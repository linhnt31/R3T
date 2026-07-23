// SPDX-License-Identifier: MIT
pragma solidity >=0.5.0 <0.9.0;

contract Contribution{
    
    struct Work {
        bool finished;
        uint dataSize;
        // we may add data quality here: 
        // uint dataQuality;
    }

    mapping(address => mapping(uint => Work)) public dataContributions;
    // The amount of data contributed by each client
    mapping(address => uint) public dataContributionsCount;
    mapping(address => uint) public balances;
    address[] public clientAddresses;
    // Round number
    uint[] public rNos;
    event contributeEvent(uint _rNo, address _address);

    function calculateContribution(uint _rNo, bool _finished, uint _dataSize, uint _payment)  public 
    {
        // The require keyword: ensure that certain conditions are met before proceeding with the execution of a function
        require(
            (dataContributions[msg.sender][_rNo].finished == true || dataContributions[msg.sender][_rNo].finished == false)
            ,
            "The client already contributed."
        );

        if (_finished == true) {
            if(_dataSize > 50){
                dataContributions[msg.sender][_rNo] = Work(_finished, _dataSize);
                balances[msg.sender] = balances[msg.sender] + _payment;
                clientAddresses.push(msg.sender);
                rNos.push(_rNo);
            }
            dataContributionsCount[msg.sender] =  dataContributionsCount[msg.sender] + _dataSize;                
        }

       emit contributeEvent(_rNo, msg.sender);
    }

    function getClientAddresses() view public returns (address[] memory){
        return clientAddresses;
    }

    function get_rNos() view public returns (uint[] memory){
        return rNos;
    }

    function getContribution(address _clientAddress, uint _rNo) view public returns (bool finished_, uint dataSize_, uint balance_){
        finished_ = dataContributions[_clientAddress][_rNo].finished;
        dataSize_ = dataContributions[_clientAddress][_rNo].dataSize;
        balance_ = balances[_clientAddress];

        return (finished_, dataSize_, balance_);
    }

    function getBalance(address _clientAddress) public view returns(uint balance_){
       return balances[_clientAddress];
    }


}
