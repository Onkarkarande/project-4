pragma solidity ^0.5.5;
contract Bank
{
int bal;
constructor() public
{
bal=0;
}
function getbalance() view public returns(int){
return bal;
}
function withdraw(int amt) public{
bal=bal-amt;
}
function deposite(int amt) public{
bal= bal+amt;
}
}