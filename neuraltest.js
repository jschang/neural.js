var $N = require('./neural').$N;

var createAndTrainWordTimeSeriesNetwork = function(words) {
    
    var network = $N.constructors.fullyConnected([26,26,words.length]);
    for(var id in network.neurons) {
        network.neurons[id].activator = $N.activators.sigmoid;
    }
    var lo = network.lo = 0.0;
    var hi = network.hi = 1.0;
    
    network.createSampleFromWord = function(word) {
        
        var network = this;
        var inputs = network.getInputs();
        var outputIds = network.samples.outputIdsByWord;
        var letters = network.samples.inputIdsByLetter;
        var thisSample = {};
        
        // zero out the sample
        for(var id in inputs) {
            thisSample[id] = lo;
        }
        for(var id in outputs) {
            thisSample[id] = lo;
        }

        for(var j=0; j<word.length; j++) {
            //console.log(word[j]);
            // lower each by half
            for(var id in inputs) {
                if(thisSample[id]>lo) {
                    thisSample[id] = .75 * thisSample[id];
                }
            }
            // assign the letter input to hi
            thisSample[ letters[ word[j] ] ] = hi;
        }
        thisSample[ outputIds[ word ] ] = hi;
        
        return thisSample;
    }
    
    network.addOutputWord = function(word) {
        var layer = this.getNextLayer(this.getOutputs(),'inputs');
        var newOut = this.neuron();
        for(var nId in layer) {
            this.synapse(this.neurons[nId],newOut);
        }
        this.samples.outputIdsByWord[word] = newOut.id;
        return newOut;
    }
    
    network.samples = network.sampler('shuffler');
    network.samples.samples = [];
    
    // assign each input to a letter
    var inputs = network.getInputs();
    var letters = {};
    var j=0;
    var a = ("a".charCodeAt(0));
    for(var id in inputs) {
        letters[ String.fromCharCode( a + j ) ] = id;
        j++;
    }
    network.samples.inputIdsByLetter = letters;
    
    // associated each word to an output
    var outputs = network.getOutputs();
    var outputIds = {};
    var i = 0;
    for(var id in outputs) {
        outputIds[words[i]] = id;
        i++;
        if(i>words.length) break;
    }
    network.samples.outputIdsByWord = outputIds;
    
    // generate and input/output sample for each word
    network.samples.samplesByWord = {};
    for(var i=0; i<words.length; i++) {
        var word = words[i];
        var sample = network.createSampleFromWord(words[i]);
        network.samples.samples.push(sample);
        network.samples.samplesByWord[word] = sample;
    }
    // turn logging off, for training
    var ol = network.log;
    network.log = function(m) {};
    // swap out the default gradient
    var og = $N.defaults.gradient;
    $N.defaults.gradient = $N.gradients.annealing(1.0,0.05,10,0.7,.00001);
    // train the network
    network.train(network.samples);
    // restore the gradient and logging
    $N.defaults.gradient = og;
    network.log = ol;
    return network;
}
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
    for(var id in net.neurons) {
        net.neurons[id].activator = $N.activators.tanh;
    }
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
    $N.utils.assertTrue($N.utils.closeTo(totalMse,0.0060510320644548725));
},
'trainingXOR':function() {
    
    var net = utils.net();
    var output = net.neuron();
    net.synapse(net.output,output);
    net.synapse(net.drain,output);
    net.output = output;
    for(var id in net.neurons) {
        net.neurons[id].activator = $N.activators.tanh;
    }
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
    $N.utils.assertTrue(net.train(samples));
    console.log(net.dataOnly());
    for(var i = 0; i<samples.samples.length; i++ ) {
        var runData = net.forward(samples.samples[i]);
        console.log('sample: '+JSON.stringify(samples.samples[i]));
        console.log('output: '+runData.activations[net.output.id]);
    }
},
'validatePreserveNetworkWithGeneratedSample':function() {
    // tests the theory that a randomized sample set
    // run through a feed forward network, 
    // to generate sample set outputs,
    // and then used as the supplementary sample data,
    // when adding a new sample to the training set,
    // will preserve the original data in the network
    var words = [
        "pants",
        "taco",
        "apple",
        "atrocity",
        "banana",
        "borogroves",
        "butter",
        "facetious",
        "galaxy",
        "harmony",
        "mimsy",
        "space",
        "philanthropy"
    ];
    var network = createAndTrainWordTimeSeriesNetwork(words);
    
    network.log = function(m) {};
        
    // create a randomized sample set, using feedforward to fill in output values
    var newOutput = network.addOutputWord("parsimony");
    var outputs = network.getOutputs();
    var inputs = network.getInputs();
    var newSamples = [];
    //for(var i=0; i < (Object.keys(network.synapses).length + Object.keys(network.neurons).length); i++) {
    for(var i=0; i < network.samples.samples.length*10; i++) {
        var newSample = {};
        for(var inId in inputs) {
            newSample[inId] = Math.random();
        }
        var runData = network.forward(newSample);
        for(var outId in outputs) {
            newSample[outId] = 
                outId == newOutput.id ? network.lo : 
                runData.activations[outId];
        }
        // exclude the new output from back prop training
        newSample.exclude = {};
        newSample.exclude[newOutput.id]=newOutput;
        newSamples.push(newSample);
    }
    var newSample = network.createSampleFromWord("parsimony");
    newSamples.push(newSample);
    var oldSamples = network.samples.samples;
    network.samples.samples = newSamples;
    
    // train with the new samples
    // turn logging off, for training
    var ol = network.log;
    network.log = function(m) {};
    // swap out the default gradient
    var og = $N.defaults.gradient;
    $N.defaults.gradient = $N.gradients.annealing(1.0,0.05,10,0.7,.00001);
    // train the network
    network.train(network.samples);
    // restore the gradient and logging
    $N.defaults.gradient = og;
    network.log = ol;
    
    // restore old samples, adding the new
    network.samples.samples = oldSamples;
    for(var i=0;i<oldSamples.length;i++) {
        network.samples.samples[i][newOutput.id] = network.lo;
    }
    network.samples.samples.push(newSample);
    //console.log(network.samples.samples);
    
    // train the network
    var trainData = network.trainData();
    trainData.samples = network.samples;
    console.log("TOTAL MSE, post: "+network.totalMse(trainData,true));
},
'validateComparativeFunction':function() {
    // validate that a network can be trained,
    // with 2 inputs and 1 output
    // such that when the 2 inputs are within a threshold delta from
    // then activation on the output is high,
    // else the activation of the output is low
},
'validateSimilarActivations':{
    // validate that when the network is trained
    // to recognize two samples with similar values
    // on a specific subset of inputs
    // that synapses and/or neurons exhibit
    // similar responses in those regions
    // this should be useful in learning how to split networks
    // and how to recognize new features
    // and as extra-net outputs to recurrent networks
},
'validateTimeSeries':function() {
    // when each input represents a discrete event
    // and the input value is allowed to degrade over time
    // can the output represent another discrete event?
    // for example, if my inputs represent each letter of the alphabet
    // and the outputs represent the known dictionary words
    // then can an input of {"a":0.6,"e":1.0,"l":0.9,"p":0.8} yield the output "apple"?
    // "a" is the first, so more decayed
    // "p" is repeated twice, thus it's distance from "a"
    // "e" is experienced last, thus the most fresh activation
    var words = [
        "apple",
        "atrocity",
        "banana",
        "borogroves",
        "butter",
        "facetious",
        "galaxy",
        "harmony",
        "mimsy",
        "space",  
        "taco",
        "target",
        "toast", // 356
        "tricky", // 654
        "philanthropy", // 362
        "phillistine", // 356
        "night", // 532
        "day", // 20000
        //"morning",
        //"evening","afternoon",
        //"one",
        //"two",
        //"three","four","five","six","seven","eight","nine","ten","zero"
    ];
    var network = createAndTrainWordTimeSeriesNetwork(words);
    var runData = network.forward(network.samples.samplesByWord.philanthropy);
    console.log(runData);
}
};

try {
    tests['everythingForward']();
    tests['trainingXOR']();
    tests['validateTimeSeries']();
    tests['validatePreserveNetworkWithGeneratedSample']();
} catch(e) {
    console.log(e.msg);
    console.log(e.stack);
}
