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
    utils:{
        closeTo:function(a,b,epsilon) {
            var e = (typeof(epsilon)==='undefined') ? .000000000001 : epsilon;
            if( Math.abs(a-b) < e ) {
                return true;
            }
            return false;
        },
        assertTrue:function(a) {
            if(typeof(a)=='undefined' || !a) {
                throw new this.exception(-1,"Assertion failed");
            }
        },
        exception:function(code,msg) {
            this.code = code;
            this.msg = msg;
        }
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
        iterator:function() {
            var ret = {
                samples:[],
                idx:0,
                next:function(){
                    if(this.samples.length==this.idx) {
                        return null;
                    }
                    return this.samples[this.idx++];
                },
                reset:function(){
                    this.idx = 0;
                }
            };
            return ret;
        },
        shuffler:function() {
            var ret = this.iterator();
            ret._shuffler_reset = ret.reset;
            ret.reset = function() {
                this._shuffler_reset();
                this.samples = this.shuffle(this.samples);
            }
            ret.shuffle = function(o){
                for(var j, x, i = o.length; i; j = Math.floor(Math.random() * i), x = o[--i], o[i] = o[j], o[j] = x);
                return o;
            }
            return ret;
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
                 * Activate forwards or backwards depending on the weightKey
                 */
                activate:function(weightKey,runData) {
                    this.log("neuron activate("+weightKey+") entering for "+this.id);
                    var inputs       = weightKey=='forwardWeight' ? this.inputs        : this.outputs;
                    // when activating using forwardWeight, pull the synapses output neuron's activation
                    var sKey         = weightKey=='forwardWeight' ? 'input'            : 'output';
                    // when activating forwardWeight, use the forwardThreshold of the neuron
                    var thresholdKey = weightKey=='forwardWeight' ? 'forwardThreshold' : 'backwardThreshold';
                    var sum = 0.0;
                    // iterate over the synapses
                    for(idS in inputs) {
                        var s = inputs[idS];
                        var thisVal = s[weightKey] * runData.activations[ s[sKey].id ];
                        this.log("neuron activate("+weightKey+") - synapse " + s.id + " " + s.str(weightKey) + " " + s[weightKey] + " * " + runData.activations[s[sKey].id] + " = " + thisVal);
                        sum += thisVal;
                    }
                    this.log('neuron activate('+weightKey+') - ' + this.id + " activation sum = " + sum);
                    // set the activation for this 
                    runData.activations[this.id] = this.activator.calculate(sum) - this[thresholdKey];
                    this.log('neuron activate('+weightKey+') - ' + this.id + " "+this.activator.calculate(sum)+" - "+this[thresholdKey]+" = activation = " + runData.activations[this.id]);
                    return runData.activations[this.id];
                },
                // when forward weights, error is calculated from output to input
                // when backward weights, error is calculated from input to output
                // when forward weights, the synapse input is the channel, to travel backwards
                // when backward weights, the synapse output is the channel, to travel forwards
                error:function(weightKey,trainData) {
                    var runData = trainData.runData;
                    this.log("neuron error("+weightKey+") entering");
                    var inputs    = weightKey=='forwardWeight'? this.outputs      : this.input;
                    var sKey      = weightKey=='forwardWeight'? 'output'          : 'input';
                    var threshKey = weightKey=='forwardWeight'? 'forwardThreshold':'backwardThreshold';
                    var sum = 0.0;
                    // iterate over the input synapses to this neuron, for the target weight
                    for(idS in inputs) {
                        var s = inputs[idS];
                        var thisVal = s[weightKey] * trainData.errors[ s[sKey].id ];
                        this.log("neuron error("+weightKey+") - "+this.id+" "
                            +"synapse "+s.str(weightKey)
                            +' wgt:'+s[weightKey]+" * err:"+trainData.errors[s[sKey].id]+" = "+thisVal);
                        sum += thisVal;
                    }
                    this.log("neuron error("+weightKey+") - " + this.id + " error sum = " + sum);
                    // set the error for this 
                    trainData.errors[this.id] = this.activator.derivative(
                        this.activator.inverse(
                            runData.activations[this.id] + this[threshKey])) * sum;
                    this.log("neuron error("+weightKey+") - "+this.id+' '
                        +'activ.deriv(activ.inv('
                        +'act:'+runData.activations[this.id]+' + thresh:' + this[threshKey] + ')) '
                        +'* sum:'+sum+' = error:' +trainData.errors[this.id]);
                    return trainData.errors[this.id];
                },
                // when adjusting forward weights, run from input to output
                // when adjusting backward weights, run from output to input
                weights:function(weightKey,trainData) {
                    var runData = trainData.runData;
                    var inputs  = weightKey=='forwardWeight'?this.inputs :this.outputs;
                    var outputs = weightKey=='forwardWeight'?this.outputs:this.inputs;
                    this.log('neuron weights('+weightKey+') - '+this.id+' update weight; '+weightKey);
                    for(idS in inputs) {
                        var s = inputs[idS];
                        var old = s[weightKey];
                        s[weightKey] = old + (trainData.errors[this.id] * runData.activations[this.id] * trainData.rate);
                        this.log('neuron weights('+weightKey+') - synapse '+idS+' '+s.str(weightKey)+' weight: '+old+' = '+s[weightKey]+' + ('+trainData.errors[this.id]+' * '+runData.activations[this.id]+' * '+trainData.rate+')');
                    }
                },
                thresholds:function(weightKey,trainData) {
                    var runData = trainData.runData;
                    var inputs  = weightKey=='forwardWeight'?this.inputs :this.outputs;
                    var outputs = weightKey=='forwardWeight'?this.outputs:this.inputs;
                    var thresholdKey = weightKey=='forwardWeight'?'forwardThreshold':'backwardThreshold';
                    this.log('neuron thresholds('+weightKey+') - '+this.id+' update threshold; '+thresholdKey);
                    var old = this[thresholdKey];
                    this[thresholdKey] = old + (trainData.errors[this.id] * runData.activations[this.id] * trainData.rate);
                    this.log('neuron thresholds('+weightKey+') - '+this.id
                        +' thres:'+this[thresholdKey]+' = '
                        +old+' - (err:'+trainData.errors[this.id]+' * act:'+runData.activations[this.id]+' * rate:'+trainData.rate+')')
                },
                activator:neuraljs.defaults.activator,
                log:function(msg) { this.network.log(msg); }
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
                    if(typeof(weightKey)=='undefined') {
                        return this.id+'('+(typeof(this.input)!='undefined'?this.input.id:'null')
                            +'-'
                            +(typeof(this.output)!='undefined'?this.output.id:'null')+')';
                    }
                    return this.id+'('+(typeof(this.input)!='undefined'?this.input.id:'null')
                        +(weightKey=='forwardWeight'?'->':'<-')
                        +(typeof(this.output)!='undefined'?this.output.id:'null')+')';
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
        o.runData = function(sample) {
            return {
                // the by-id activation of each neuron in the network
                activations:{},
                direction:'outputs',
                weightKey:'forwardWeight',
                sample:sample
            };
        }
        o.trainData = function(runData) {
            var net = this;
            return {
                network:net,
                runData:runData,
                // the by-id error of each neuron
                errors:{},
                // the slope of input value at each neuron
                slopes:{},
                weightKey:'forwardWeight',
                thresholdKey:'forwardThreshold',
                // the training rate
                rate:.02,
                // the source of input vectors
                samples:{
                    next:function(){return null;},
                    reset:function(){}
                },
                mse:function() {
                    var outputs = this.network[this.weightKey=='forwardWeight'?'getOutputs':'getInputs']();
                    var mse=0;
                    var i=0.0;
                    for(nId in outputs) {
                        mse += Math.pow(this.errors[nId],2)
                        i += 1.0;
                    }
                    return mse / i;
                }
            };
        }
        o.add = function(input) {
            this.neurons[input.id] = input;
        }
        o.run = function(startLayerNeurons, direction, neuronFunction, neuronArgs) {
            if(!startLayerNeurons) {
                return;
            }
            this.log("network run("+direction+") - running with: "+JSON.stringify(Object.keys(startLayerNeurons)));
            var i = 0;
            for(id in startLayerNeurons) {
                var neuron = startLayerNeurons[id];
                neuronFunction.apply(neuron,neuronArgs);
            }
            this.run(this.getNextLayer(startLayerNeurons,direction), direction, neuronFunction, neuronArgs);
        }
        o.getNextLayer = function(startLayerNeurons,direction) {
            this.log("network getNextLayer("+direction+") - running with: "+JSON.stringify(Object.keys(startLayerNeurons)));
            var nextLayerNeurons = {};
            var nextLayerConn = direction == 'outputs' ? 'output' : 'input';
            var i = 0;
            for(id in startLayerNeurons) {
                var neuron = startLayerNeurons[id];
                // foreach synapse in the direction we're going
                for(sId in neuron[direction]) {
                    // check if there is an output neuron
                    if(typeof(neuron[direction][sId][nextLayerConn])==='undefined') {
                        continue;
                    }
                    // if there is an output neuron, add it to the next layer
                    var nextNeuron = neuron[direction][sId][nextLayerConn];
                    nextLayerNeurons[nextNeuron.id] = nextNeuron;
                }
                i++;
            }
            if(i==0) {
                return null;
            }
            return nextLayerNeurons;
        }
        o.forward = function(actualData) {
            this.log('network forward() entering');
            var runData = this.runData(actualData);
            // fill in the runData activations for the input neurons
            for(id in this.getInputs()) {
                this.log("network forward() - input neuron "+id+" = "+actualData[id]);
                runData.activations[id] = actualData[id];
            }
            o.run(
                    this.getNextLayer(this.getInputs(),'outputs')
                    ,'outputs'
                    ,function(){this.activate('forwardWeight',runData);}
                    ,[]
                );
            return runData;
        }
        o.backward = function(actualData) {
            this.log('network backward() entering');
            var runData = this.runData();
            runData.direction = 'inputs';
            runData.weightKey = 'backwardWeight';
            // fill in the runData activations for the output neurons
            for(id in this.getOutputs()) {
                this.log("network backward() - output neuron "+id+" = "+actualData[id]);
                runData.activations[id] = actualData[id];
            }
            o.run(
                    this.getNextLayer(this.getOutputs(),'inputs')
                    ,'inputs'
                    ,function(){this.activate('backwardWeight',runData);}
                    ,[]
                );
            return runData;
        }
        o.weights = function(weightKey,trainData) {
            this.log('network weights('+weightKey+') entering');
            var endNeurons = this[weightKey=='forwardWeight'?'getInputs':'getOutputs']();
            o.run(  endNeurons
                    ,weightKey=='forwardWeight'?'outputs':'inputs'
                    ,function(){this.weights(weightKey,trainData);}
                    ,[]
                );
        }
        o.thresholds = function(weightKey,trainData,runData) {
            this.log('network thresholds('+weightKey+') entering');
            var endNeurons = this[weightKey=='forwardWeight'?'getInputs':'getOutputs']();
            o.run(  endNeurons
                    ,weightKey=='forwardWeight'?'outputs':'inputs'
                    ,function(){this.thresholds(weightKey,trainData);}
                    ,[]
                );
        }
        o._error = function(weightKey,trainData) {
            var sumAct = 0;
            var runData = trainData.runData;
            // when adjusting the forward weights, we travel backwards from
            // the outputs...forward from the inputs when adjusting backward weights
            var endNeurons = this[weightKey=='forwardWeight'?'getOutputs':'getInputs']();
            var threshKey = weightKey=='forwardWeight'?'forwardThreshold':'backwardThreshold';
            // populate the error at the end points
            for(idN in endNeurons) {
                var n = endNeurons[idN];
                var actual = runData.activations[n.id];
                var desired = runData.sample[n.id];
                trainData.errors[n.id] = (desired) - (actual);
                trainData.slopes[n.id] = n.activator.derivative(
                    n.activator.inverse(
                            actual+n[threshKey]
                        )
                    );
                this.log('network _error('+weightKey+') - end layer error for ' + idN + " "+desired+" - "+actual+" = " + trainData.errors[n.id]);
                this.log('network _error('+weightKey+') - end layer slope for ' + idN + " derivative(inverse( "+actual+" + "+n[threshKey]+" ) = " + trainData.slopes[n.id]);
            }
            this.log(trainData);
            // walk the error backward
            var direction = weightKey=='forwardWeight'?'inputs':'outputs';
            this.run(  
                    this.getNextLayer(endNeurons,direction)
                    ,direction
                    ,function(){this.error(weightKey,trainData,runData);}
                    ,[]
                );
        }
        // backward from outputs, adjusts forward weights
        o.backwardPropError = function(trainData) {
            this.log('network backwardPropError() entering');
            trainData.weightKey = 'forwardWeight';
            trainData.thresholdKey = 'forwardThreshold';
            return this._error('forwardWeight',trainData);
        }
        // forward from inputs, adjusts backward weights
        o.forwardPropError = function(trainData) {
            this.log('network forwardPropError() entering');
            trainData.weightKey = 'backwardWeight';
            trainData.thresholdKey = 'backwardThreshold';
            return this._error('backwardWeight',trainData);
        }
        o.train = function(sampleSource) {
            var sample = null;
            while(sample = sampleSource.next()) {
                this.log(sample);
                
                this.log("+= TRAINING FORWARD =======");
                var runData = this.forward(sample);
                var trainData = this.trainData(runData);
                this.backwardPropError(trainData);
                this.weights('forwardWeight',trainData);
                this.thresholds('forwardWeight',trainData);
                this.log(trainData);
                this.log(this.neurons);
                
                this.log("+= TRAINING BACKWARD =======");
                var runData = this.backward(sample);
                var trainData = this.trainData(runData);
                this.forwardPropError(trainData);
                this.weights('backwardWeight',trainData);
                this.thresholds('backwardWeight',trainData);
                this.log(trainData);
                this.log(this.neurons);
            }
        }
        o.sampler = function(name, args) {
            return $N.samplers[name].apply(this,args);
        }
        o.log = function(msg) {
            console.log(msg);
        }
        o.dataOnly = function() {
            var pref = 'network str() - ';
            var t = {neurons:{},synapses:{}};
            for(nId in this.neurons) {
                var n = this.neurons[nId];
                t.neurons[nId] = {
                    forwardThreshold:n.forwardThreshold,
                    backwardThreshold:n.backwardThreshold
                };
            }
            for(sId in this.synapses) {
                var s = this.synapses[sId];
                t.synapses[sId] = {
                    connections:s.str(),
                    forwardWeight:s.forwardWeight,
                    backwardWeight:s.backwardWeight
                };
            }
            return t;
        }
        return o;
    }
};

module.extend(exports.neuraljs.defaults,{
    activator:neuraljs.activators.tanh
});

var $N = exports.$N = neuraljs;



