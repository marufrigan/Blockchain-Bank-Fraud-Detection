const FraudDetection = artifacts.require("FraudDetection");
module.exports = function(deployer) {
  deployer.deploy(FraudDetection);
};
