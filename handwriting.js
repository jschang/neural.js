var network = neuraljs.newNetwork();

$(document).ready(function() {
    var o = network;
    // create the input neurons
    var endIdx = $('#canvas')[0].width * $('#canvas')[0].height;
    for(var idx = 0; idx < endIdx; idx++) {
        var newN = o.newNeuron();
        o.inputs.push(newN);
        o.neurons[newN.id] = newN;
    }
    // fully connect all input neurons to the output neurons
    var outN = o.newNeuron();
    o.neurons[outN.id] = outN;
    for(var i=0; i < o.inputs.length; i++) {
        var inN = o.inputs[i];
        var newS = o.newSynapse();
        inN.addOutput(newS);
        outN.addInput(newS);
        o.synapses[newS.id] = newS;
    }
});

function canvasCoords(evt) {
    return {
        x:Math.floor((evt.pageX-evt.currentTarget.offsetLeft)*(evt.currentTarget.width/evt.currentTarget.clientWidth)),
        y:Math.floor((evt.pageY-evt.currentTarget.offsetTop)*(evt.currentTarget.height/evt.currentTarget.clientHeight)),
        equals:function(other) {
            if(typeof other.x == 'undefined' || typeof other.y == 'undefined') {
                return false;
            }
            if(other.x==this.x && other.y==this.y) {
                return true;
            }
            return false;
        },
        str:function() {
            return this.x+','+this.y;
        }
    };
}

$('#canvas').each(function() {
    var ctx = this.getContext("2d");
    ctx.fillStyle = 'white';
    ctx.fillRect(0,0,this.width,this.height);
    this.lastCoords = {x:-1,y:-1};
});
$('#canvas').mousedown(function(evt) {
    var coords = canvasCoords(evt);
    this.lastCoords = coords;
    this.mouseDown = true;
});
$('#canvas').mouseup(function(evt) {
    var coords = canvasCoords(evt);
    if(this.mouseDown) {
        var ctx = this.getContext("2d");
        ctx.fillStyle="#000000";
        ctx.fillRect(coords.x,coords.y,1,1);
    }
    this.lastCoords = coords;
    this.mouseDown = false;
});
$('#canvas').mousemove(function(evt) {
    var coords = canvasCoords(evt);
    if(coords.equals(this.lastCoords)) {
        return;
    }
    if(this.mouseDown) {
        var ctx = this.getContext("2d");
        ctx.beginPath();
        ctx.moveTo(this.lastCoords.x,this.lastCoords.y);
        ctx.lineTo(coords.x,coords.y);
        console.log("from: "+this.lastCoords.str()+' to '+coords.str());
        ctx.strokeStyle="#000000";
        ctx.stroke();
    }
    this.lastCoords = coords;
});
$('#add_sample').mouseup(function(evt) {
        
    var primaryCanvas = $('#canvas')[0];
    
    // duplicate the properties of our primary canvas
    var newSample = $('<div style="clear:both;"><canvas style="border:1px solid blue;"></canvas><div class="run_data"></div></div>')[0];
    var newSampleCanvas = $('canvas',newSample)[0];
    newSampleCanvas.width = primaryCanvas.width;
    newSampleCanvas.height = primaryCanvas.height;
    
    // normalize to our input format and attach to the new canvas for the training set
    var ctx = primaryCanvas.getContext('2d');
    var imgData = ctx.getImageData(0,0,primaryCanvas.width,primaryCanvas.height);
    var actualData = [];
    for (var i=0;i<imgData.data.length;i+=4)
    {
        if( imgData.data[i+0]==imgData.data[i+1]
            && imgData.data[i+1]==imgData.data[i+2]
            && imgData.data[i+2]==255 ) {
            actualData[i/4] = 0;
        } else actualData[i/4] = 1;
    }
    for(var i=0; i<primaryCanvas.height; i++) {
        var s = "";
        for(var j=0; j<primaryCanvas.width; j++) {
            s = s+','+actualData[(primaryCanvas.width*i)+j];
        }
        console.log(s);
    }
    newSampleCanvas.actualData = actualData;
    
    // actually fill in the new canvas with data from our primary
    var newSampleCtx = newSampleCanvas.getContext('2d');
    newSampleCtx.createImageData(primaryCanvas.width,primaryCanvas.height);
    newSampleCtx.putImageData(imgData,0,0);
    
    // tack the sample on to the end
    $('#training_set').append(newSample);
    
    // reset the primary canvas
    $('#canvas').each(function() {
        var ctx = this.getContext("2d");
        ctx.fillStyle = 'white';
        ctx.fillRect(0,0,this.width,this.height);
    });
});
