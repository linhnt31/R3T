from web3 import Web3
import json
import os
import sys
import time

from threading import Thread

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


class BlockchainService():

    def __init__(self, client_id: int = -1):
        role = f"client_{client_id}" if client_id >= 0 else "client"
        self.metrics = BlockchainMetricsCollector(role=role)
        self.client_id = client_id

    def addWeight(self, _session: int, _round_num: int, _dataSize: int,
                  _filePath: str, _fileHash: str, client_id: int):
        client_address = w3.eth.accounts[client_id + 1]
        with self.metrics.record("addWeight"):
            tx_hash = federation_contract_instance.functions.addWeight(
                _session, _round_num, _dataSize, _filePath, _fileHash
            ).transact({'from': client_address})
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        self.metrics.attach_receipt(receipt)
        # Record actual on-chain calldata bytes (ABI-encoded input), not the .npy file size
        raw_input = w3.eth.get_transaction(receipt['transactionHash'])['input']
        _cd = raw_input[2:] if raw_input.startswith('0x') else raw_input
        self.metrics.attach_extras({"calldata_bytes": len(bytes.fromhex(_cd))})
        result = federation_contract_instance.functions.getWeight(_session, _round_num).call()
        return result

    def getAddress(self, client_id: int):
        return w3.eth.accounts[client_id + 1]

    def getContributions(client_address):
        roundNumbers = contribution_contract_instance.functions.get_rNos().call()
        contributions = []
        for rNo in roundNumbers:
            contribution = contribution_contract_instance.functions.getContribution(client_address, rNo).call()
            if contribution[1] > 0:
                contrib_dic = {}
                contrib_dic[0] = contribution[0]
                contrib_dic[1] = contribution[1]
                contrib_dic[2] = contribution[2]
                contrib_dic[3] = rNo
                contributions.append(contrib_dic)
        return contributions


def handle_event(event):
    print(event)
    from client_api import FLlaunch
    FLlaunch = FLlaunch()
    FLlaunch.start()


def log_loop(event_filter, poll_interval):
    print(event_filter.get_new_entries())
    while True:
        for event in event_filter.get_new_entries():
            handle_event(event)
        time.sleep(poll_interval)


def main():
    block_filter = federation_contract_instance.events.addStrategyEvent.createFilter(fromBlock='latest')
    worker = Thread(target=log_loop, args=(block_filter, 1), daemon=True)
    worker.start()


# The event-listener is only needed when running via the FastAPI (client_api.py) flow.
# When using the CLI (client_run.py), clients connect directly — no listener required.
if __name__ == "__main__":
    main()
