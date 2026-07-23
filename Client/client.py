import argparse
import os
import hashlib
import time
import gc
import numpy as np
import random
import tensorflow as tf
import flwr as fl

from blockchain_service import *
from verify import *
from blockchain_metrics import LatencyCommLogger

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, client_id, client_address,
                 blockchain_service=None, latency_comm_logger=None):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.client_id = client_id
        self.client_address = client_address
        # Accept an externally created instance (with metrics) or fall back to a plain one
        self.blockchainService = blockchain_service if blockchain_service is not None \
            else BlockchainService(client_id=client_id)
        # Optional logger for training & blockchain latency/communication per round
        self.latency_comm_logger: LatencyCommLogger = latency_comm_logger

    def get_properties(self, config):
        """Get properties of client."""
        return {"client_id": self.client_id}

    def get_parameters(self, config):
        # """Get parameters of the local model."""
        # raise Exception("Not implemented (server-side parameter initialization)")
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        session: int = config["session"]
        session_dir: str = config.get("session_dir", f"Session-{session}")
        round: int = config["round"]
        clp: int = config["clp"]
        # Use values provided by the config
        print(f"[Client {self.client_id} round {round}] fit, config: {config}\n")
        
        # Train the model using hyperparameters from config
        # Measure communication: parameters received from server (download)
        download_bytes = sum(p.nbytes for p in parameters)

        _train_t0 = time.perf_counter()
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            # validation_split=0.2,
            validation_data=(self.x_test, self.y_test), # update to fix the bug 'WARNING:tensorflow:Your input ran out of data;...'
            # https://www.tensorflow.org/hub/tutorials/tf2_image_retraining
            steps_per_epoch=max(1, len(self.x_train) // batch_size),
            validation_steps=max(1, len(self.x_test) // batch_size)
        )
        _train_latency = time.perf_counter() - _train_t0

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        # Communication: parameters sent back to server (upload) — same shape as download
        upload_bytes = sum(p.nbytes for p in parameters_prime)

        # Compute CLP loss metric: lr * ||∇L(batch)||²
        # This measures gradient magnitude post-training, used for critical learning period detection
        x_batch = tf.constant(self.x_train[:batch_size], dtype=tf.float32)
        y_batch = tf.constant(self.y_train[:batch_size])
        with tf.GradientTape() as tape:
            preds = self.model(x_batch, training=False)
            batch_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y_batch, preds)
            )
        gradients = tape.gradient(batch_loss, self.model.trainable_variables)
        grad_norm_sq = sum(
            float(tf.reduce_sum(tf.square(g)).numpy())
            for g in gradients if g is not None
        )
        clp_loss = float(self.model.optimizer.learning_rate) * grad_norm_sq

        results = {
            "client_id": self.client_id,
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
            "client_address": self.client_address,
            "clp_loss": clp_loss,
        }

        # Save training weights in the created directory
        if not (os.path.exists(f'../Client/Local-weights')):
            os.mkdir(f"../Client/Local-weights")

        if not (os.path.exists(f'../Client/Local-weights/Client-{self.client_id}')):
            os.mkdir(f"../Client/Local-weights/Client-{self.client_id}")

        if not (os.path.exists(f'../Client/Local-weights/Client-{self.client_id}/{session_dir}')):
            os.mkdir(f"../Client/Local-weights/Client-{self.client_id}/{session_dir}")

        filename = f'../Client/Local-weights/Client-{self.client_id}/{session_dir}/Round-{round}-training-weights.npy'
        np.save(filename, parameters_prime)
        # Time the SHA-256 hash computation — this is pure blockchain-prep overhead
        _hash_t0 = time.perf_counter()
        with open(filename, "rb") as f:
            _file_bytes = f.read()
            readable_hash = hashlib.sha256(_file_bytes).hexdigest()
        _hash_latency = time.perf_counter() - _hash_t0
        print("\n=====Weights digested at {}: {}=====".format(round, readable_hash))

        # Log hash computation as a blockchain overhead event
        if self.latency_comm_logger is not None:
            self.latency_comm_logger.log(
                part="blockchain",
                event="hash_computation",
                round_num=round,
                latency_s=_hash_latency,
                # SHA-256 hexdigest = 64 hex chars = 64 ASCII bytes (on-chain string size)
                comm_bytes=len(readable_hash.encode()),
                client_id=self.client_id,
            )

        # Log training latency and communication before the blockchain call
        if self.latency_comm_logger is not None:
            self.latency_comm_logger.log(
                part="training",
                event="fit",
                round_num=round,
                latency_s=_train_latency,
                comm_bytes=download_bytes + upload_bytes,
                client_id=self.client_id,
                download_bytes=download_bytes,
                upload_bytes=upload_bytes,
                num_examples=num_examples_train,
            )

        # Client ith add weights to blockchain
        bcResult = self.blockchainService.addWeight(_session=session,_round_num=round, _dataSize=num_examples_train, _filePath = filename, _fileHash = readable_hash, client_id=self.client_id)

        # Log blockchain addWeight overhead:
        #   comm_bytes  = actual ABI-encoded calldata bytes sent to the chain node
        #   model_bytes = raw .npy file size (FL model size that drove this TX)
        if self.latency_comm_logger is not None:
            bc_txs = self.blockchainService.metrics.transactions
            bc_latency = bc_txs[-1]["latency_s"] if bc_txs else 0.0
            calldata_bytes = bc_txs[-1].get("calldata_bytes", 0) if bc_txs else 0
            model_bytes = os.path.getsize(filename) if os.path.exists(filename) else 0
            self.latency_comm_logger.log(
                part="blockchain",
                event="addWeight",
                round_num=round,
                latency_s=bc_latency,
                comm_bytes=calldata_bytes,
                model_bytes=model_bytes,
                client_id=self.client_id,
            )

        # Release temporary tensors/objects created in this round.
        del history, x_batch, y_batch, gradients
        gc.collect()

        # Here, we use the number of data points for calculate contribution
        # So, we need to send them to blockchain
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)
        session: int = config["session"]
        session_dir: str = config.get("session_dir", f"Session-{session}")
        round: int = config["round"]

        # Get config values
        steps: int = config["val_steps"]

        # Get global weights
        global_rnd_model = self.model.get_weights()

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)

        # Create directory for global weights
        try:
            if not (os.path.exists(f'../Client/Global-weights')):
                os.mkdir(f"../Client/Global-weights")

            if not (os.path.exists(f'../Client/Global-weights/{session_dir}')):
                os.mkdir(f"../Client/Global-weights/{session_dir}")

            filename = f'../Client/Global-weights/{session_dir}/Round-{round}-Global-weights.npy'
            if not (os.path.exists(filename)):
                np.save(filename, global_rnd_model)
        except NameError:
            print(NameError)
        finally:
            del global_rnd_model
            gc.collect()

        return loss, num_examples_test, {"accuracy": accuracy}
