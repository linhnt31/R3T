from web3 import Web3
import json
import math
import os
import sys
from decimal import Decimal

# Shared metrics collector
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blockchain_metrics import BlockchainMetricsCollector

SEED = 345
DATASET_NAME = 'cifar10'
rpcServer = 'HTTP://127.0.0.1:8545'

w3 = Web3(Web3.HTTPProvider(rpcServer))

# Resolve the live network ID so we don't rely on a hardcoded '5777'
_network_id = str(w3.net.version)

contributionSC = open('../Blockchain/build/contracts/Contribution.json')
contributionData = json.load(contributionSC)
contributionAbi = contributionData['abi']
addressContribution = contributionData['networks'][_network_id]['address']
contribution_contract_instance = w3.eth.contract(address=addressContribution, abi=contributionAbi)

federationSC = open('../Blockchain/build/contracts/Federation.json')
federationData = json.load(federationSC)
federationAbi = federationData['abi']
addressFederation = federationData['networks'][_network_id]['address']
federation_contract_instance = w3.eth.contract(address=addressFederation, abi=federationAbi)

event_filter = contribution_contract_instance.events.contributeEvent.createFilter(fromBlock='latest')
poll_interval = 2


class BlockchainService():

    def __init__(self):
        self.metrics = BlockchainMetricsCollector(role="server")

    def getContributions(self):
        clientAddresses = contribution_contract_instance.functions.getClientAddresses().call()
        rNos = contribution_contract_instance.functions.get_rNos().call()
        results = []
        for client in clientAddresses:
            for round in rNos:
                contribution = contribution_contract_instance.functions.getContribution(client, round).call()
                contrib_dic = {}
                contrib_dic[0] = contribution[0]
                contrib_dic[1] = contribution[1]
                contrib_dic[2] = contribution[2]
                contrib_dic[3] = round
                contrib_dic[4] = client
                results.append(contrib_dic)
        return results

    def addStrategy(self, _session: int, _algoName: str, _numRounds: int, _numClients: int):
        server_account = w3.eth.accounts[0]
        with self.metrics.record("addStrategy"):
            tx_hash = federation_contract_instance.functions.addStrategy(
                _session, _algoName, _numRounds, _numClients
            ).transact({'from': server_account})
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        self.metrics.attach_receipt(receipt)
        # Capture actual ABI-encoded calldata bytes sent to the chain node.
        # This reflects true blockchain communication overhead (not model size).
        raw_input = w3.eth.get_transaction(receipt['transactionHash'])['input']
        _cd = raw_input[2:] if raw_input.startswith('0x') else raw_input
        self.metrics.attach_extras({"calldata_bytes": len(bytes.fromhex(_cd))})
        strategy = federation_contract_instance.functions.getStrategy(_session).call()
        return strategy

    def addContribution(self, _rNo: int, _dataSize: int, _client_address: str,
                        _totalDataSize: int, _totalBudget: int,
                        number_of_rounds: int, client_id: int, joined_round: list):
        server_account = w3.eth.accounts[0]

        payment = Decimal("0")
        with open('config_training.json', 'r') as config_training:
            data = json.loads(config_training.read())
            clp = data['clp']

        for r in joined_round:
            if r < clp:
                factor = Decimal(str(1 + 2 / math.log(2 * r)))
                payment += factor * factor  # based on Eq. 3 and Eq. 9
            else:
                payment += Decimal("1")

        # EVM/Ganache values should be integer-based on chain.
        payment_eth = float(payment)
        payment_for_contract = int(round(payment_eth))
        payment_wei = w3.toWei(payment, 'ether')

        balance_before = w3.eth.getBalance(_client_address)

        # 1. Record the contribution state update on-chain (calculateContribution TX)
        with self.metrics.record("addContribution"):
            tx_hash = contribution_contract_instance.functions.calculateContribution(
                _rNo, True, _dataSize, payment_for_contract
            ).transact({"from": _client_address})
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        self.metrics.attach_receipt(receipt)
        # Capture actual ABI-encoded calldata bytes for the contribution TX
        raw_input = w3.eth.get_transaction(receipt['transactionHash'])['input']
        _cd = raw_input[2:] if raw_input.startswith('0x') else raw_input
        self.metrics.attach_extras({
            "calldata_bytes":  len(bytes.fromhex(_cd)),
            "client_address":  _client_address,
        })

        # 2. Record the ETH payment transfer as a separate TX so its latency and
        #    gas are captured independently from the contribution state update.
        with self.metrics.record("paymentTransfer"):
            pay_hash = w3.eth.send_transaction({
                'from': server_account,
                'to':   _client_address,
                'value': payment_wei,
            })
            pay_receipt = w3.eth.wait_for_transaction_receipt(pay_hash)
        self.metrics.attach_receipt(pay_receipt)

        balance_after = w3.eth.getBalance(_client_address)
        self.metrics.attach_extras({
            "payment_eth":        payment_eth,
            "payment_int":        payment_for_contract,
            "client_address":     _client_address,
            "balance_before_wei": balance_before,
            "balance_after_wei":  balance_after,
        })

        contribution = contribution_contract_instance.functions.getContribution(_client_address, _rNo)
        return contribution

    def addModel(self, _session: int, _round_num: int, _filePath: str, _fileHash: str):
        server_account = w3.eth.accounts[0]
        with self.metrics.record("addModel"):
            tx_hash = federation_contract_instance.functions.addModel(
                _session, _round_num, _filePath, _fileHash
            ).transact({'from': server_account})
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        self.metrics.attach_receipt(receipt)
        # Capture actual ABI-encoded calldata bytes sent to the chain node
        raw_input = w3.eth.get_transaction(receipt['transactionHash'])['input']
        _cd = raw_input[2:] if raw_input.startswith('0x') else raw_input
        self.metrics.attach_extras({"calldata_bytes": len(bytes.fromhex(_cd))})
        model = federation_contract_instance.functions.getModel(_session, _round_num).call()
        return model

    def getModel(self, _session: int, _round_num: int):
        return federation_contract_instance.functions.getModel(_session, _round_num).call()

    def getTrainingSessions(self):
        clientAddresses = contribution_contract_instance.functions.getClientAddresses().call()
        rNos = contribution_contract_instance.functions.get_rNos().call()
        results = []
        for round in rNos:
            session_dict = {}
            dataSize = 0
            contributors = 0
            for client in clientAddresses:
                contribution = contribution_contract_instance.functions.getContribution(client, round).call()
                if contribution[1] > 0:
                    dataSize += int(contribution[1])
                    contributors += 1
            session_dict["roundNumber"] = round
            session_dict["contributors"] = contributors
            session_dict["dataSize"] = dataSize
            results.append(session_dict)
        return results

    def snapshot_client_balances(self, contribution: dict, label: str) -> None:
        """
        Query the on-chain ETH balance for every client in *contribution* and
        store a named snapshot in the metrics collector.

        Parameters
        ----------
        contribution : dict
            The ``strategy.contribution`` dict from FLstrategy (keys are client IDs
            plus the special ``"total_data_size"`` key).
        label : str
            Snapshot label, e.g. ``"before_payment"`` or ``"after_payment"``.
        """
        balances = {}
        for client_id, info in contribution.items():
            if client_id == 'total_data_size':
                continue
            addr = info['client_address']
            balances[f"client_{client_id}"] = {
                "address":     addr,
                "balance_wei": w3.eth.getBalance(addr),
            }
        self.metrics.record_balances(label, balances)

    def getLastModel(self, _session: int, _round_num: int):
        return federation_contract_instance.functions.getModel(_session, _round_num).call()
