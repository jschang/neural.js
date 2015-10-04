var $N = require('./neural').$N;

var net = $N.network();

var input1 = net.neuron();
var input2 = net.neuron();
var output = net.neuron();
var drain = net.neuron();

net.synapse(input1,output);
net.synapse(input2,output);
net.synapse(input1,drain);
net.synapse(input2,drain);

var samples = $N.samplers.shuffler();
var rawData = [
    {input1:1.0,input2:1.0,output:0.0,drain:1.0}
    ,{input1:0.0,input2:1.0,output:1.0,drain:0.0}
    ,{input1:1.0,input2:0.0,output:1.0,drain:0.0}
    ,{input1:0.0,input2:0.0,output:0.0,drain:1.0}
];
for(var i=0; i<rawData.length; i++) {
    var o = {};
    o[input1.id] = rawData[i].input1;
    o[input2.id] = rawData[i].input2;
    o[output.id] = rawData[i].output;
    o[drain.id] = rawData[i].drain;
    samples.samples.push(o);
}

var trainData = net.trainData();
trainData.samples = samples;
console.log(samples.samples[0]);

var runData = net.forward(samples.samples[0]);
console.log(runData);
net.error(trainData,runData,samples.samples[0]);
console.log(trainData);

//var runData = net.backward(samples.samples[0]);
//console.log(runData);
