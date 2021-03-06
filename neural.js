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
        assertNumber:function(n) {
            if(n+''=='NaN') {
                throw new this.exception(-1,"Assertion failed, variable is NaN");
            }
            return n;
        },
        clamp:function(v,l,h) {
            if(v>0 && h < v) {
                return h;
            }
            if(v<0 && l > v) {
                return l;
            }
            return v;
        },
        exception:function(code,msg) {
            this.code = code;
            this.msg = msg;
            this.stack = new Error().stack;
        },
        undef:function(s) {
            var ret = typeof(s)!=='undefined';
            //console.log(ret+" = typeof(s)!=='undefined'");
            return ret;
        }
    },
    activators:{
        tanh:{
            name:'tanh',
            calculate:function(x) {
                //return Math.tanh(x);
                var num   = Math.pow(Math.E,x)-Math.pow(Math.E,-x);
                var denom = Math.pow(Math.E,x)+Math.pow(Math.E,-x);
                return num / denom;
            },
            derivative:function(x) {
                //x = $N.utils.clamp(x,-.9999999999999999,.9999999999999999);
                return 1.0 - Math.pow(this.calculate(x),2.0);
            }
        },
        sigmoid:{
            name:'sigmoid',
            calculate:function(x) {
                //x = $N.utils.clamp(x,.0000000001,9999999999);
                return 1.0 / ( 1.0 + Math.pow(Math.E,-x) );
            },
            derivative:function(x) {
                return this.calculate(x) * (1.0 - this.calculate(x));
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
    acceptors:{
        simple:function() {
            return function() { 
                return this.lastTotalMse===null 
                    || this.totalMse < this.lastTotalMse;
            }
        }
    },
    gradients:{
        simple:function(constRate) {
            return function() {
                return constRate;
            };
        },
        annealing:function(maxRate,minRate,bumpDist,dropRate,minDiff) {
            var lastSuccessfulRate = null;
            var lastRate = null;
            return function() {
                var logMe = function (k) {
                    //console.log(k+' mnr:'+minRate+',mxr:'+maxRate+',lr:'+lastRate+',cr:'+this.currentRate+',lsr:'+lastSuccessfulRate+',md:'+minDiff);
                }
                var i = 0;
                if (lastRate==null) {
                    lastSuccessfulRate = this.currentRate = maxRate;
                    logMe.apply(this,[i++]);
                }
                else if (this.runsSinceLastAccept==0 && minDiff<(this.totalMse-this.lastTotalMse)) {
                    lastSuccessfulRate = this.currentRate;
                    this.currentRate = lastSuccessfulRate;
                    logMe.apply(this,[i++]);
                }
                else if (this.runsSinceLastAccept>bumpDist) {
                    if(this.currentRate>=minRate) {
                        lastRate = this.currentRate;
                        this.currentRate *= dropRate;
                        logMe.apply(this,[i++]);
                    } else if (this.currentRate<minRate && this.currentRate != maxRate) {
                        lastRate = this.currentRate;
                        this.currentRate = lastSuccessfulRate;
                        logMe.apply(this,[i++]);
                    }
                }
                
                if(lastRate!=this.currentRate) {
                    this.runsSinceLastGradientChange = 0;
                } else {
                    this.runsSinceLastGradientChange ++;
                }
                if(lastSuccessfulRate<minRate) {
                    lastSuccessfulRate=maxRate;
                    logMe.apply(this,[i++]);
                }
                lastRate = this.currentRate;
                
                return this.currentRate;
            }
        }
    },
    completers:{
        simple:function(threshold,bailCount) {
            return function() {
                if(''+this.totalMse=='NaN') {
                    this.achievedConvergence = false;
                    return true;
                }
                if(this.runs>bailCount) {
                    this.achievedConvergence = false;
                    return true;
                }
                if(this.totalMse < threshold) {
                    this.achievedConvergence = true;
                    return true;
                }
                return false;
            }
        }
    },
    /**
     * Training object constructors
     * Each training object function is applied against a trainingData object
     */
    trainers:{
        regular:function() {
            return {
                complete:$N.defaults.completer,
                gradient:$N.defaults.gradient,
                accept:$N.defaults.acceptor,
                train:function() {
                    
                    var net = this.network;
                    var weightKey = this.weightKey;
                    var newRate = this.network.trainer.gradient.apply(this);
                    var samples = this.samples;
                    
                    do {
                        
                        // make a copy of the network
                        var clonedNet = net.cloneAll();
                        if(false) {
                            console.log('ORIGINAL');
                            console.log(net.dataOnly());
                            console.log('CLONE');
                            console.log(clonedNet.dataOnly());
                        }
                        
                        // do a training run
                        while( (thisSample = samples.next()) != null ) {

                            var thisRunData = clonedNet[weightKey=='forwardWeight'?'forward':'backward'](thisSample);
                            
                            var thisTrainData = clonedNet.trainData(thisRunData);
                            thisTrainData.weightKey = this.weightKey;
                            thisTrainData.thresholdKey = this.thresholdKey;
                            thisTrainData.currentRate = newRate;
                            
                            clonedNet[weightKey=='forwardWeight'?'backwardPropError':'forwardPropError'](thisTrainData);
                            clonedNet.weights(weightKey,thisTrainData);
                            clonedNet.thresholds(weightKey,thisTrainData);
                        }
                        samples.reset();
                        
                        // evaluate what we just did
                        clonedNet.totalMse(this);
                        
                        // if the totalMse is less than the lastTotalMse
                        if( this.network.trainer.accept.apply(this) ) {
                            // then overwrite the original network data with the recently trained
                            net.copyFrom(clonedNet);
                            this.lastTotalMse = this.totalMse;
                            this.runsSinceLastAccept = 0;
                            console.log(this.runs+" New acceptance at Total Error: "+this.totalMse+', gradient:'+newRate);
                        } else {
                            this.runsSinceLastAccept++;
                        }
                        this.runs++;
                        newRate = this.network.trainer.gradient.apply(this);
                        if(this.runs%10==0) {
                            console.log(this.runs+","+this.runsSinceLastAccept+" Total Error: "+this.totalMse+', gradient:'+newRate);
                        }
                        
                    } while( ! this.network.trainer.complete.apply(this) );
                }
            };
        }
    },
    constructors:{
        fullyConnected:function(layerCounts) {
            var layers = [];
            var network = $N.network();
            for(var i=0; i<layerCounts.length; i++) {
                var thisLayer = [];
                for(var k=0; k<layerCounts[i]; k++) {
                    var neuron = network.neuron();
                    thisLayer.push(neuron);
                }
                layers.push(thisLayer);
            }
            var previousLayer = layers[0];
            for(var i=1; i<layerCounts.length; i++) {
                var thisLayer = layers[i];
                for(var k=0; k<thisLayer.length; k++) {
                    for(var j=0; j<previousLayer.length; j++) {
                        network.synapse(previousLayer[j],thisLayer[k]);
                    }
                }
                previousLayer = thisLayer;
            }
            return network;
        }
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
            synapses:{}, // connections between inputs and other neurons
            trainer:$N.trainers.regular()
        };
        o.neuron = function(id) {
            var net = this;
            var n = {
                id:typeof(id)!=='undefined'?id:net.lastId++,
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
                        
                        sum += thisVal;
                    }
                    this.log('neuron activate('+weightKey+') - ' + this.id + " activation sum = " + sum);
                    
                    // set the activation for this 
                    var sum = sum + this[thresholdKey];
                    runData.sums[this.id] = sum;
                    runData.activations[this.id] = this.activator.calculate(sum);
                    
                    return $N.utils.assertNumber(runData.activations[this.id]);
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
                        
                        // the input synapses weight multiplied by the connecting neuron's error
                        var thisVal = s[weightKey] * trainData.errors[ s[sKey].id ];
                        
                        sum += thisVal;
                    }

                    // set the error for this 
                    trainData.errors[this.id] = sum;
                    
                    return $N.utils.assertNumber(trainData.errors[this.id]);
                },
                // when adjusting forward weights, run from input to output
                // when adjusting backward weights, run from output to input
                weights:function(weightKey,trainData) {
                    var runData = trainData.runData;
                    var inputs  = weightKey=='forwardWeight'?this.inputs :this.outputs;
                    var outputs = weightKey=='forwardWeight'?this.outputs:this.inputs;
                    var sKey    = weightKey=='forwardWeight'?'input':'output';
                    
                    this.log('neuron weights('+weightKey+') - '+this.id+' update weight; '+weightKey);
                    
                    for(idS in inputs) {
                        var s = inputs[idS];
                        var sN = this.network.neurons[s[sKey].id];
                        
                        var old = s[weightKey];
                        s[weightKey] = $N.utils.assertNumber(
                            s[weightKey] + (
                                this.activator.derivative(runData.sums[this.id])
                                * runData.activations[sN.id]
                                * trainData.errors[this.id] 
                                * trainData.currentRate
                            )
                        );
                    }
                },
                thresholds:function(weightKey,trainData) {
                    var runData = trainData.runData;
                    var inputs  = weightKey=='forwardWeight'?this.inputs :this.outputs;
                    var outputs = weightKey=='forwardWeight'?this.outputs:this.inputs;
                    var thresholdKey = weightKey=='forwardWeight'?'forwardThreshold':'backwardThreshold';
                    
                    this.log('neuron thresholds('+weightKey+') - '+this.id+' update threshold; '+thresholdKey);
                    
                    var old = this[thresholdKey];
                    this[thresholdKey] = $N.utils.assertNumber(
                        this[thresholdKey] + (
                            this.activator.derivative(runData.sums[this.id])
                            * trainData.errors[this.id]
                            * trainData.currentRate
                        )
                    );
                },
                activator:neuraljs.defaults.activator,
                log:function(msg) { this.network.log(msg); },
                data:function() {
                    var ret = {};
                    ret.id = this.id;
                    ret.forwardThreshold = this.forwardThreshold;
                    ret.backwardThreshold = this.backwardThreshold;
                    ret.outputs = {};
                    for(var id in this.outputs) {
                        ret.outputs[id] = this.outputs[id].data();
                    }
                    return ret;
                }
            };
            this.neurons[n.id] = n;
            return n;
        }
        o.endLayerError = function(trainData) {
            var runData = trainData.runData;
            var weightKey = trainData.weightKey;
            // when adjusting the forward weights, we travel backwards from
            // the outputs...forward from the inputs when adjusting backward weights
            var endNeurons = this[weightKey=='forwardWeight'?'getOutputs':'getInputs']();
            var threshKey = weightKey=='forwardWeight'?'forwardThreshold':'backwardThreshold';
            // populate the error at the end points
            for(idN in endNeurons) {
                var n = endNeurons[idN];
                var actual = runData.activations[n.id];
                var desired = runData.sample[n.id];
                
                trainData.errors[n.id] = desired - actual;

                var oldThresh = n[threshKey];
                n[threshKey] = n[threshKey] + ( 
                        n.activator.derivative(runData.sums[n.id])
                        * trainData.errors[n.id] 
                        * trainData.currentRate 
                    );
            }
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
        o.synapse = function(inputOrId,output) {
            var input, id;
            if(typeof(inputOrId)!=='undefined' && typeof(inputOrId)!=='object') {
                id = inputOrId;
            } else if(typeof(inputOrId)!=='undefined') {
                input = inputOrId;
            }
            var net = this;
            var s = {
                id:typeof(id)!=='undefined'?id:net.lastId++,
                input:null,
                output:null,
                forwardWeight:Math.random(),
                backwardWeight:Math.random(),
                network:net,
                data:function() {
                    var ret = {};
                    ret.forwardWeight = this.forwardWeight;
                    ret.backwardWeight = this.backwardWeight;
                    ret.outputId = this.output ? this.output.id : null;
                    return ret;
                },
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
                sums:{},
                direction:'outputs',
                weightKey:'forwardWeight',
                sample:sample
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
                runData.sums[id] = actualData[id];
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
            var runData = this.runData(actualData);
            runData.direction = 'inputs';
            runData.weightKey = 'backwardWeight';
            // fill in the runData activations for the output neurons
            for(id in this.getOutputs()) {
                this.log("network backward() - output neuron "+id+" = "+actualData[id]);
                runData.activations[id] = actualData[id];
                runData.sums[id] = actualData[id];
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
            // when adjusting the forward weights, we travel backwards from
            // the outputs...forward from the inputs when adjusting backward weights
            var endNeurons = this[weightKey=='forwardWeight'?'getOutputs':'getInputs']();
            for(var id in trainData.samples.exclude) {
                if(typeof(endNeurons[id]!=='undefined')) {
                    delete endNeurons[id];
                }
            }
            // walk the error backward
            var direction = weightKey=='forwardWeight'?'inputs':'outputs';
            this.run(  
                    this.getNextLayer(endNeurons,direction)
                    ,direction
                    ,function(){this.error(weightKey,trainData);}
                    ,[]
                );
        }
        // backward from outputs, adjusts forward weights
        o.backwardPropError = function(trainData) {
            this.log('network backwardPropError() entering');
            trainData.weightKey = 'forwardWeight';
            trainData.thresholdKey = 'forwardThreshold';
            this.endLayerError(trainData);
            return this._error('forwardWeight',trainData);
        }
        // forward from inputs, adjusts backward weights
        o.forwardPropError = function(trainData) {
            this.log('network forwardPropError() entering');
            trainData.weightKey = 'backwardWeight';
            trainData.thresholdKey = 'backwardThreshold';
            this.endLayerError(trainData);
            return this._error('backwardWeight',trainData);
        }
        o.trainData = function(runData) {
            var net = this;
            var trainData = {
                
                /* transient training data */
                // the sample being evaluated
                thisSample:null,
                // the totalMse of this candidate
                totalMse:0, 
                sampleCount:0,
                // the current lowest mse
                lastTotalMse:null,
                runs:0,
                // the number of runs since last accepted candidate clone
                runsSinceLastAccept:0,
                runsSinceLastGradientChange:0,
                // whether the training run ultimately was successful
                achievedConvergence:false, 
                currentRate:0.05,
                
                network:net,
                runData:runData,
                // the by-id error of each neuron
                errors:{},
                weightKey:'forwardWeight',
                thresholdKey:'forwardThreshold',

                // the source of input vectors
                samples:{
                    next:function(){return null;},
                    reset:function(){}
                },
                
                mse:function() {
                    var outputs = this.network[this.weightKey=='forwardWeight'?'getOutputs':'getInputs']();
                    var mse=0.0;
                    var i=0.0;
                    for(nId in outputs) {
                        mse += Math.pow(this.runData.sample[nId] - this.runData.activations[nId],2)
                        i++;
                    }
                    return mse/i;
                }
            };
            return trainData;
        }
        o.train = function(samples,weightKey) {
            var goodSession = false;
            // train input to output
            var clonedNet = this.cloneAll();
            var trainData = clonedNet.trainData();
            if(typeof(weightKey)!='undefined') {
                trainData.weightKey = weightKey;
                trainData.thresholdKey = weightKey=='forwardWeight'?'forwardThreshold':'backwardThreshold';
            }
            trainData.samples = samples;
            clonedNet.trainer.train.apply(trainData);
            if(trainData.achievedConvergence) {
                this.copyFrom(clonedNet);
                goodSession = true;
            }
            return goodSession;
        }
        o.totalMse = function(trainData,log) {
            trainData.sampleCount = 0.0;
            trainData.totalMse = 0.0;
            while( (trainData.thisSample = trainData.samples.next()) != null ) {
            
                var thisRunData = this
                    [trainData.weightKey=='forwardWeight'?'forward':'backward']
                    (trainData.thisSample);
                if(typeof(log)!='undefined') {
                    var outputs = {};
                    for(var id in this.getOutputs()) {
                        outputs[id] = thisRunData.activations[id];
                    }
                    console.log(outputs);
                }
                var thisTrainData = this.trainData(thisRunData);
                thisTrainData.weightKey = trainData.weightKey;
                thisTrainData.thresholdKey = trainData.thresholdKey;
                
                trainData.totalMse += Math.pow(thisTrainData.mse(),.5);
                trainData.sampleCount += 1.0;
            }
            trainData.samples.reset();
            
            return trainData.totalMse = trainData.totalMse / trainData.sampleCount;
        }
        o.sampler = function(name, args) {
            return $N.samplers[name].apply($N.samplers,args);
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
        o.data = function() {
            var ret = {};
            ret.inputIds = Object.keys(this.getInputs());
            return ret;
        }
        o.cloneAll = function() {
            var r = $N.network();
            for(var i in this) {
                if(typeof(this[i])=='function') {
                    r[i] = this[i];
                }
            }
            for(var nId in this.neurons) {
                var n = r.neuron(nId);
                var thisN = this.neurons[nId];
                n.activator = thisN.activator;
                n.forwardThreshold = thisN.forwardThreshold;
                n.backwardThreshold = thisN.backwardThreshold;
            }
            for(var sId in this.synapses) {
                var s = r.synapse(sId);
                var thisS = this.synapses[sId];
                s.forwardWeight = thisS.forwardWeight;
                s.backwardWeight = thisS.backwardWeight;
                if(typeof(thisS.input)!=='undefined') {
                    var rN = r.neurons[thisS.input.id];
                    s.input = rN;
                    rN.outputs[s.id] = s;
                }
                if(typeof(thisS.output)!=='undefined') {
                    var rN = r.neurons[thisS.output.id];
                    s.output = rN;
                    rN.inputs[s.id] = s;
                }
            }
            r.lastId = this.lastId;
            return r;
        }
        o.copyFrom = function(r) {
            for(var nId in this.neurons) {
                var n = r.neurons[nId];
                var thisN = this.neurons[nId];
                thisN.forwardThreshold = n.forwardThreshold;
                thisN.backwardThreshold = n.backwardThreshold;
            }
            for(var sId in this.synapses) {
                var s = r.synapses[sId];
                var thisS = this.synapses[sId];
                thisS.forwardWeight = s.forwardWeight;
                thisS.backwardWeight = s.backwardWeight;
            }
            this.lastId = r.lastId;
        }
        o.save = function(name) {
            $.ajax('neural.php',{
                method:'POST',
                data:{
                    db:'neuraljs',
                    col:'networks',
                    key:name,
                    op:'update',
                    doc:JSON.stringify(network.data())
                },
                success:function(data) {
                    console.log('success');
                    console.log(data);
                }
            });
            for(var id in network.neurons) {
                $.ajax('neural.php',{
                    method:'POST',
                    data:{
                        db:'neuraljs',
                        col:'neurons',
                        key:name+'_'+id,
                        op:'update',
                        doc:JSON.stringify(network.neurons[id].data())
                    },
                    success:function(data) {
                        console.log('success');
                        console.log(data);
                    }
                });  
            }
        }
        return o;
    }
};
var $N = exports.$N = neuraljs;
module.extend(exports.neuraljs.defaults,{
    activator:$N.activators.tanh,
    gradient:$N.gradients.simple(.05),
    acceptor:$N.acceptors.simple(),
    completer:$N.completers.simple(.05,20000)
});



