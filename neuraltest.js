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
    {input1:0.90,input2:0.90,output:0.10,drain:0.90}
    ,{input1:0.10,input2:0.90,output:0.90,drain:0.10}
    ,{input1:0.90,input2:0.10,output:0.90,drain:0.10}
    ,{input1:0.10,input2:0.10,output:0.10,drain:0.90}
];
for(var i=0; i<rawData.length; i++) {
    var o = {};
    o[input1.id] = rawData[i].input1;
    o[input2.id] = rawData[i].input2;
    o[output.id] = rawData[i].output;
    o[drain.id] = rawData[i].drain;
    samples.samples.push(o);
}

net.train(samples);
