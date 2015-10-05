/*
priorities:
- maintain separation of runData and trainData from networkData.
*/

var module = this;
if(typeof(exports)=='undefined') {
    var exports = {};
}
module.ifNode = function(_if, _else) {
    if(typeof(window)=='undefined') {
        if(typeof(_if)!='undefined') {
            _if();
        }
    } else {
        console.log('not in node');
        if(typeof(_else)!='undefined') {
            _else();
        }
    }
}
module.ifNode(
    function(){module.extend = require('extend');},
    function(){module.extend = $.extend;}
);

var neuraljs = exports.neuraljs = {
    // this object is flushed out at the bottom of the file
    defaults:{
        activator:null
    },
    activators:{
        tanh:{
            calculate:function(x) {
                var num   = Math.pow(Math.E,x)-Math.pow(Math.E,-x);
                var denom = Math.pow(Math.E,x)+Math.pow(Math.E,-x);
                return num / denom;
            },
            derivative:function(x) {
                return 1.0 - Math.pow(this.calculate(x),2.0);
            },
            inverse:function(x) {
                return .5 * ( (Math.log(1.0+x,Math.E)) - (Math.log(1.0-x,Math.E)) );
            }
        }
        /*
        gaussian elimination javascript available at
        http://martin-thoma.com/solving-linear-equations-with-gaussian-elimination/#tocAnchor-1-4
        given:
            w1*a1 + w2*a2 + w3+a3 = k1
                    w4*a2 + w5+a3 = k2
        where w4 and w5 are new synapses, in preparation for feed backwards
        it's probably ok if w4 and w5 are randomized, initial values,
        so long as one of the training set resulting in k1 is passed 
        through to determine k2 prior to feeding backward from k1.
        they can still be useful in determining the activations.
        of course, that would only match that specific sample.
        why not train backwards and forwards?
        */
    },
    samplers:{
        shuffler:function() {
            return {
                samples:[],
                idx:0,
                next:function(){
                    if(this.samples.length==this.idx) {
                        return null;
                    }
                    return this.samples[this.idx++];
                },
                reset:function(){
                    this.samples = this.shuffle(this.samples);
                    this.idx = 0;
                },
                shuffle:function(o){
                    for(var j, x, i = o.length; i; j = Math.floor(Math.random() * i), x = o[--i], o[i] = o[j], o[j] = x);
                    return o;
                }
            };
        }
    },
    network:function() {
        var o = {
            lastId:0,
            neurons:{}, // the neurons, including inputs, of this network
            synapses:{} // connections between inputs and other neurons
        };
        o.neuron = function() {
            var net = this;
            var n = {
                id:net.lastId++,
                network:net,
                forwardThreshold:Math.random(),
                backwardThreshold:Math.random(),
                outputs:{}, // output synapses
                inputs:{},  // input synapses
                addInput:function(inS) {
                    this.inputs[inS.id] = inS;
                    inS.output = this;
                },
                addOutput:function(outS) {
                    this.outputs[outS.id] = outS;
                    outS.input = this;
                },
                /**
                 * Walk forwards or backwards depending on the weightKey
                 */
                run:function(weightKey,neuronFunction,neuronFunctionInputs) {
                    var inputs  = weightKey=='forwardWeight'?this.inputs :this.outputs;
                    var outputs = weightKey=='forwardWeight'?this.outputs:this.inputs;
                    var sKey    = weightKey=='forwardWeight'?'input' :'output';
                    var invSKey = weightKey=='forwardWeight'?'output':'input';
                    this.log('neuron '+this.id+' neuronFunctions');
                    // activate each in the next layer
                    for(idS in outputs) {
                        var s = outputs[idS];
                        neuronFunction.apply(s[invSKey],neuronFunctionInputs);
                    }
                    // run forward each in this layer
                    this.log('neuron '+this.id+' runs');
                    for(idS in inputs) {
                        var s = inputs[idS];
                        s[sKey].run(weightKey,neuronFunction,neuronFunctionInputs);
                    }
                },
                /**
                 * Activate forwards or backwards depending on the weightKey
                 */
                activate:function(weightKey,runData) {
                    var inputs  = weightKey=='forwardWeight'?this.inputs :this.outputs;
                    var outputs = weightKey=='forwardWeight'?this.outputs:this.inputs;
                    var sKey =    weightKey=='forwardWeight'?'input':'output';
                    var thresholdKey = weightKey=='forwardWeight'?'forwardThreshold':'backwardThreshold';
                    var sum = 0.0;
                    // if we've already done this one, then just return
                    if(typeof runData.activations[this.id] != 'undefined') {
                        this.log('neuron ' + this.id + ' activation already set');
                        return runData.activations[this.id];
                    }
                    // iterate over the synapses
                    for(idS in inputs) {
                        var s = inputs[idS];
                        var thisVal = s[weightKey] * runData.activations[ s[sKey].id ];
                        this.log("synapse " + s.id + " " + s[weightKey] + " * runData.activations[" + s[sKey].id + "] = " + thisVal);
                        sum += thisVal;
                    }
                    this.log('neuron ' + this.id + " activation sum = " + sum);
                    // set the activation for this 
                    runData.activations[this.id] = this.activator.calculate(sum) - this[thresholdKey];
                    return runData.activations[this.id];
                },
                /**
                 * Calculate error forwards or backwards depending on the weightKey.
                 * If weight key is forward, then the backward weights are updated
                 * When weight key is backward, then the forward weights are updated
                 */
                error:function(weightKey,trainData,runData) {
                    //this.log(runData);
                    var inputs  = weightKey=='forwardWeight'?this.inputs :this.outputs;
                    var outputs = weightKey=='forwardWeight'?this.outputs:this.inputs;
                    var sKey =    weightKey=='forwardWeight'?'input':'output';
                    var invWeightKey = weightKey=='forwardWeight'?'backwardWeight':'forwardWeight';
                    var invThresholdKey = weightKey=='forwardWeight'?'backwardThreshold':'forwardThreshold';
                    var sum = 0.0;
                    // if we've already done this one, then just return
                    if(typeof trainData.errors[this.id] != 'undefined') {
                        this.log(this.id + ' error already set; '+invWeightKey);
                        return trainData.errors[this.id];
                    }
                    // iterate over the synapses
                    for(idS in inputs) {
                        var s = inputs[idS];
                        var thisVal = s[invWeightKey] * trainData.errors[ s[sKey].id ];
                        this.log("synapse " + s.id + ' '+s.str(invWeightKey)+' w(' + s[invWeightKey] + ") * e(" + trainData.errors[ s[sKey].id ] + ") = " + thisVal);
                        sum += thisVal;
                    }
                    this.log(this.id + " error sum = " + sum + '; '+ invWeightKey);
                    // set the activation for this 
                    trainData.errors[this.id] = this.activator.derivative(
                        this.activator.inverse(runData.activations[this.id]+this[invThresholdKey])) * sum;
                    this.log('neuron '+this.id+' this.activator.derivative(this.activator.inverse('+runData.activations[this.id]+')) * '+sum+' = ' +trainData.errors[this.id]);
                    return trainData.errors[this.id];
                },
                /**
                 * If weight key is forward, then the backward weights are updated
                 * When weight key is backward, then the forward weights are updated
                 */
                weights:function(weightKey,trainData,runData) {
                    var inputs  = weightKey=='forwardWeight'?this.inputs :this.outputs;
                    var outputs = weightKey=='forwardWeight'?this.outputs:this.inputs;
                    var invWeightKey = weightKey=='forwardWeight'?'backwardWeight':'forwardWeight';
                    this.log('neuron '+this.id+' update weight; '+weightKey);
                    for(idS in inputs) {
                        var s = inputs[idS];
                        var old = s[weightKey];
                        s[weightKey] = old + (trainData.errors[this.id] * runData.activations[this.id] * trainData.rate);
                        this.log('synapse '+idS+' '+s.str(weightKey)+' weight: '+old+' = '+s[weightKey]+' + ('+trainData.errors[this.id]+' * '+runData.activations[this.id]+' * '+trainData.rate+')');
                    }
                },
                thresholds:function(weightKey,trainData,runData) {
                    var inputs  = weightKey=='forwardWeight'?this.inputs :this.outputs;
                    var outputs = weightKey=='forwardWeight'?this.outputs:this.inputs;
                    var thresholdKey = weightKey=='forwardWeight'?'forwardThreshold':'backwardThreshold';
                    this.log('neuron '+this.id+' update threshold; '+thresholdKey);
                    var old = this[thresholdKey];
                    this[thresholdKey] = old - (trainData.errors[this.id] * runData.activations[this.id] * trainData.rate);
                    this.log('neuron '+this.id+' threshold: '+this[thresholdKey]+' = '+old+' - ('+trainData.errors[this.id]+' * '+runData.activations[this.id]+' * '+trainData.rate+')')
                },
                activator:neuraljs.defaults.activator,
                log:function(msg) { return; this.network.log(msg); }
            };
            this.add(n);
            return n;
        }
        o.getOutputs = function() {
            var outputs = {};
            for(id in this.neurons) {
                if(Object.keys(this.neurons[id].outputs).length==0) {
                    outputs[id] = this.neurons[id];
                } 
            }
            return outputs;
        }
        o.getInputs = function() {
            var inputs = {};
            for(id in this.neurons) {
                if(Object.keys(this.neurons[id].inputs).length==0) {
                    inputs[id] = this.neurons[id];
                } 
            }
            return inputs;
        }
        o.synapse = function(input,output) {
            var net = this;
            var s = {
                id:net.lastId++,
                input:null,
                output:null,
                forwardWeight:Math.random(),
                backwardWeight:Math.random(),
                network:net,
                str:function(weightKey) {
                    return (typeof(this.input)!='undefined'?this.input.id:'null')
                        +(weightKey=='forwardWeight'?'->':'<-')
                        +(typeof(this.output)!='undefined'?this.output.id:'null');
                }
            }
            if(typeof(input)!='undefined') {
                input.addOutput(s);
            }
            if(typeof(output)!='undefined') {
                output.addInput(s);
            }
            this.synapses[s.id] = s;
            return s;
        }
        o.runData = function() {
            return {
                // the by-id activation of each neuron in the network
                activations:{}
            };
        }
        o.trainData = function() {
            return {
                // the by-id error of each neuron
                errors:{},
                // the slope of input value at each neuron
                slopes:{},
                // the training rate
                rate:.02,
                // the source of input vectors
                samples:{
                    next:function(){return null;},
                    reset:function(){}
                }
            };
        }
        o.add = function(input) {
            this.neurons[input.id] = input;
        }
        o.forward = function(actualData) {
            this.log('network forward() entering');
            var runData = this.runData();
            // fill in the runData activations for the input neurons
            for(id in this.getInputs()) {
                this.log("input neuron "+id+" = "+actualData[id]);
                runData.activations[id] = actualData[id];
            }
            // feed forward activate the network
            for(id in this.getInputs()) {
                var n = this.neurons[id];
                n.run('forwardWeight'
                    ,function(){this.activate('forwardWeight',runData);}
                    ,[]);
            }
            return runData;
        }
        o.backward = function(actualData) {
            this.log('network backward() entering');
            var runData = this.runData();
            // fill in the runData activations for the input neurons
            for(id in this.getOutputs()) {
                this.log("output neuron "+id+" = "+actualData[id]);
                runData.activations[id] = actualData[id];
            }
            // feed forward activate the network
            for(id in this.getOutputs()) {
                var n = this.neurons[id];
                n.run('backwardWeight'
                    ,function(){this.activate('backwardWeight',runData);},
                    []);
            }
            return runData;
        }
        o.train = function(sampleSource) {
            var sample = null;
            while(sample = sampleSource.next()) {
                this.log(sample);
                var trainData = this.trainData();
                var runData = this.forward(sample);
                this.backwardError(trainData,runData,sample);
                this.weights('forwardWeight',trainData,runData);
                this.thresholds('forwardWeight',trainData,runData);
                this.log(trainData);
                
                var trainData = this.trainData();
                var runData = this.backward(sample);
                this.forwardError(trainData,runData,sample);
                this.weights('backwardWeight',trainData,runData);
                this.thresholds('backwardWeight',trainData,runData);
                this.log(trainData);
            }
        }
        o.backwardError = function(trainData,runData,desiredInputs) {
            this.log('network backwardError() entering');
            return this._error('backwardWeight',trainData,runData,desiredInputs);
        }
        o.forwardError = function(trainData,runData,desiredOutputs) {
            this.log('network forwardError() entering');
            return this._error('forwardWeight',trainData,runData,desiredOutputs);
        }
        o._error = function(weightKey,trainData,runData,desiredValues) {
            var sumAct = 0;
            var endNeurons = this[weightKey=='forwardWeight'?'getInputs':'getOutputs']();
            // populate the error at the end points
            for(idN in endNeurons) {
                this.log('find end layer error for ' + idN);
                var n = endNeurons[idN];
                var actual = runData.activations[n.id];
                var desired = desiredValues[n.id];
                trainData.errors[n.id] = desired - actual;
                trainData.slopes[n.id] = n.activator.derivative(
                    n.activator.inverse(actual));
            }
            this.log(trainData);
            // walk the error backward
            for(idN in endNeurons) {
                this.log("error run for end neuron "+idN);
                var n = endNeurons[idN];
                n.run(weightKey
                    ,function(){this.error(weightKey,trainData,runData)}
                    ,[]);
            }
        }
        o.weights = function(weightKey,trainData,runData) {
            this.log('network weights() entering');
            var endNeurons = this[weightKey=='forwardWeight'?'getInputs':'getOutputs']();
            // walk the error backward
            for(idN in endNeurons) {
                this.log("running for end neuron "+idN+'; '+weightKey);
                var n = endNeurons[idN];
                n.run(weightKey
                    ,function(){this.weights(weightKey,trainData,runData)}
                    ,[]);
            }
        }
        o.thresholds = function(weightKey,trainData,runData) {
            this.log('network thresholds() entering');
            var endNeurons = this[weightKey=='forwardWeight'?'getInputs':'getOutputs']();
            // walk the error backward
            for(idN in endNeurons) {
                this.log("running for end neuron "+idN+'; '+weightKey);
                var n = endNeurons[idN];
                n.run(weightKey
                    ,function(){this.thresholds(weightKey,trainData,runData)}
                    ,[]);
            }
        }
        o.log = function(msg) {
            console.log(msg);
        }
        return o;
    }
};

module.extend(exports.neuraljs.defaults,{
    activator:neuraljs.activators.tanh
});

var $N = exports.$N = neuraljs;



