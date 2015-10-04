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
                threshold:Math.random(),
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
                run:function(runData,weightKey) {
                    var inputs  = weightKey=='forwardWeight'?this.inputs :this.outputs;
                    var outputs = weightKey=='forwardWeight'?this.outputs:this.inputs;
                    var sKey    = weightKey=='forwardWeight'?'input' :'output';
                    var invSKey = weightKey=='forwardWeight'?'output':'input';
                    this.network.log('activate');
                    // activate each in the next layer
                    for(idS in outputs) {
                        var s = outputs[idS];
                        s[invSKey].activate(runData,weightKey);
                    }
                    // run forward each in this layer
                    this.network.log('runs');
                    for(idS in inputs) {
                        var s = inputs[idS];
                        s[sKey].run(runData,weightKey);
                    }
                },
                activate:function(runData,weightKey) {
                    var inputs  = weightKey=='forwardWeight'?this.inputs :this.outputs;
                    var outputs = weightKey=='forwardWeight'?this.outputs:this.inputs;
                    var sKey = weightKey=='forwardWeight'?'input':'output';
                    var sum = 0.0;
                    // if we've already done this one, then just return
                    if(typeof runData.activations[this.id] != 'undefined') {
                        this.network.log(this.id + ' activation already set');
                        return runData.activations[this.id];
                    }
                    // iterate over the synapses
                    for(idS in inputs) {
                        var s = inputs[idS];
                        var thisVal = s[weightKey] * runData.activations[ s[sKey].id ];
                        this.network.log("synapse " + s.id + " " + s[weightKey] + " * runData.activations[" + s[sKey].id + "] = " + thisVal);
                        sum += thisVal;
                    }
                    this.network.log(this.id + " sum = " + sum);
                    // set the activation for this 
                    runData.activations[this.id] = this.activator.calculate(sum) - this.threshold;
                    return runData.activations[this.id];
                },
                activator:neuraljs.defaults.activator
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
                network:net
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
                rate:1.0,
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
            var runData = this.runData();
            // fill in the runData activations for the input neurons
            for(id in this.getInputs()) {
                this.log("input neuron "+id+" = "+actualData[id]);
                runData.activations[id] = actualData[id];
            }
            // feed forward activate the network
            for(id in this.getInputs()) {
                var n = this.neurons[id];
                n.run(runData,'forwardWeight');
            }
            return runData;
        }
        o.backward = function(actualData) {
            var runData = this.runData();
            // fill in the runData activations for the input neurons
            for(id in this.getOutputs()) {
                this.log("output neuron "+id+" = "+actualData[id]);
                runData.activations[id] = actualData[id];
            }
            // feed forward activate the network
            for(id in this.getOutputs()) {
                var n = this.neurons[id];
                n.run(runData,'backwardWeight');
            }
            return runData;
        }
        o.train = function(sampleSource) {
            var trainData = this.trainData();
            trainData.sampleSource = sampleSource;
        }
        o.error = function(trainData,runData,desiredOutputs) {
            var sumAct = 0;
            var outputs = this.getOutputs();
            for(outIdN in outputs) {
                var outN = outputs[outIdN];
                var actual = runData.activations[outN.id];
                var desired = desiredOutputs[outN.id];
                trainData.errors[outN.id] = desired - actual;
                trainData.slopes[outN.id] = outN.activator.derivative(
                    outN.activator.inverse(actual));
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



