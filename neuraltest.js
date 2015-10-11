var $N = require('./neural').$N;

var utils = {
    'net':function() {
        
        var net = $N.network();
    
        net.input1 = net.neuron();
        net.input2 = net.neuron();
        net.output = net.neuron();
        net.drain = net.neuron();
        
        net.synapse(net.input1,net.output);
        net.synapse(net.input2,net.output);
        net.synapse(net.input1,net.drain);
        net.synapse(net.input2,net.drain);
        
        var k = 0.1;
        var i = 1.0;
        for(sId in net.synapses) {
            net.synapses[sId].forwardWeight = k*i++;
            net.synapses[sId].backwardWeight = k*i++;
        }
        var i = 1.0;
        for(nId in net.neurons) {
            net.neurons[nId].forwardThreshold = k*i++;
            net.neurons[nId].backwardThreshold = k*i++;
        }
        return net;
    },
    'sampler':function(net,rawData) {
        var ret = net.sampler('iterator');
        for(var i=0; i<rawData.length; i++) {
            var o = {};
            o[net.input1.id] = rawData[i].input1;
            o[net.input2.id] = rawData[i].input2;
            o[net.output.id] = rawData[i].output;
            o[net.drain.id] = rawData[i].drain;
            ret.samples.push(o);
        }
        return ret;
    }
};
var tests = { 
'everythingForward':function() {
    // create network
    var net = utils.net();
    net.log = function(m) {};
    
    var rawData = [
        {input1:0.10,input2:0.20,output:0.30,drain:0.40},
        {input1:0.50,input2:0.60,output:0.70,drain:0.80},
        {input1:0.10,input2:0.10,output:0.10,drain:0.10},
    ];
    var samples = utils.sampler(net,rawData);
    
    // verify run forward result
    var runData = net.forward(samples.samples[0]);
    //console.log(runData);
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.output.id],-0.430114109683571));
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.drain.id],-0.5122537941317147));
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.input1.id],.1));
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.input2.id],.2));
    var runData = net.forward(samples.samples[1]);
    //console.log(runData);
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.output.id],-0.27397164772132904));
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.drain.id],-0.11502011711927118));
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.input1.id],0.5));
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.input2.id],0.6));
    var runData = net.forward(samples.samples[2]);
    //console.log(runData);
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.output.id],-0.4600213196888364));
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.drain.id],-0.5805727014656141));
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.input1.id],.1));
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.input2.id],.1));
    
    var runData = net.forward(samples.samples[0]);
    var trainData = net.trainData(runData);
    net.backwardPropError(trainData);
    var expected = { 
            errors:{ 
                '0': 0.5079727757128458,
                '1': 0.6432089165979538,
                '2': 0.730114109683571,
                '3': 0.9122537941317147 
            },
            slopes: { 
                '2': 0.99511596233468, 
                '3': 0.9647513621820634 
            },
        };
    for(var i in expected) {
        for(var k in expected[i]) {
            //console.log("checking expected["+i+"]["+k+"]==trainData["+i+"]["+k+"]");
            $N.utils.assertTrue($N.utils.closeTo(expected[i][k],trainData[i][k]));
        }
    }
    net.weights('forwardWeight',trainData);
    net.thresholds('forwardWeight',trainData);
    
    var thisSample;
    var totalMse, sampleCount, runs=0;
    do {
        totalMse = 0; sampleCount = 0;
        while( (thisSample = samples.next()) != null ) {
            var runData = net.forward(thisSample);
            var trainData = net.trainData(runData);
            net.backwardPropError(trainData);
            net.weights('forwardWeight',trainData);
            net.thresholds('forwardWeight',trainData);
            totalMse += trainData.mse();
            sampleCount += 1;
        }
        totalMse = totalMse / sampleCount;
        //console.log(runs+" Total MSE: "+totalMse);
        samples.reset();
        runs++;
    } while( runs!=258 );
    $N.utils.assertTrue($N.utils.closeTo(totalMse,1501.6790799162063));
},
'everythingBackward':function() {
},
'trainingXOR':function() {
    var net = utils.net();
    
    var rawData = [
        {input1:0.90,input2:0.90,output:0.10,drain:0.10}
        ,{input1:0.10,input2:0.90,output:0.90,drain:0.10}
        ,{input1:0.90,input2:0.10,output:0.90,drain:0.10}
        ,{input1:0.10,input2:0.10,output:0.10,drain:0.10}
    ];
    var samples = utils.sampler(net,rawData);
    console.log(samples);
    var runData = net.forward(samples.samples[0]);
    console.log(runData);
}};

tests['everythingForward']();
//tests['trainingXOR']();
