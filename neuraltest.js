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
    'sampler':function(net,rawData,sampler) {
        var s = typeof(sampler)!='undefined'?sampler:'iterator';
        var ret = net.sampler(s);
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
    //console.log(net.dataOnly());
    
    var rawData = [
        {input1:0.10,input2:0.20,output:0.30,drain:0.40},
        {input1:0.50,input2:0.60,output:0.70,drain:0.80},
        {input1:0.10,input2:0.10,output:0.10,drain:0.10},
    ];
    var samples = utils.sampler(net,rawData);
    
    // verify run forward result
    var runData = net.forward(samples.samples[0]);
    //console.log(runData);
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.output.id],0.5153592780074097));
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.drain.id],0.7113937318189625));
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.input1.id],.1));
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.input2.id],.2));
    var runData = net.forward(samples.samples[1]);
    //console.log(runData);
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.output.id],0.623065349572361));
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.drain.id],0.8786921933686959));
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.input1.id],0.5));
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.input2.id],0.6));
    var runData = net.forward(samples.samples[2]);
    //console.log(runData);
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.output.id],0.49298796667532446));
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.drain.id],0.6750698748386078));
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.input1.id],.1));
    $N.utils.assertTrue($N.utils.closeTo(runData.activations[net.input2.id],.1));
    
    var runData = net.forward(samples.samples[0]);
    var trainData = net.trainData(runData);
    net.backwardPropError(trainData);
    var expected = { 
            errors:{ 
                '0': -0.17723279371022221,
                '1': -0.2825833956754967,
                '2': -0.21535927800740967,
                '3': -0.31139373181896246 
            }
        };
    //console.log(trainData);
    for(var i in expected) {
        for(var k in expected[i]) {
            //console.log("checking expected["+i+"]["+k+"]==trainData["+i+"]["+k+"]");
            $N.utils.assertTrue($N.utils.closeTo(expected[i][k],trainData[i][k]));
        }
    }
    net.weights('forwardWeight',trainData);
    net.thresholds('forwardWeight',trainData);
    
    var thisSample;
    var lastTotalMse = null, totalMse = null, sampleCount, runs=0;
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
    $N.utils.assertTrue($N.utils.closeTo(totalMse,0.011045160132405677));
},
'everythingBackward':function() {
},
'trainingXOR':function() {
    var net = utils.net();
    var output = net.neuron();
    net.synapse(net.output,output);
    net.synapse(net.drain,output);
    net.output = output;
    net.log = function() {};
    var lo = -1, hi = 1;
    var rawData = [
        {input1:hi,input2:hi,output:lo}
        ,{input1:lo,input2:hi,output:hi}
        ,{input1:hi,input2:lo,output:hi}
        ,{input1:lo,input2:lo,output:lo}
    ];
    var samples = utils.sampler(net,rawData,'shuffler');
    for(var i=0; i<samples.samples.length; i++) {
        delete(samples.samples[i][net.drain.id]);
    }
    net.train(samples);
    for(var i = 0; i<samples.samples.length; i++ ) {
        var runData = net.forward(samples.samples[i]);
        console.log('sample: '+JSON.stringify(samples.samples[i]));
        console.log('output: '+runData.activations[net.output.id]);
    }
}};

try {
    tests['everythingForward']();
    tests['trainingXOR']();
} catch(e) {
    console.log(e.msg);
    console.log(e.stack);
}
