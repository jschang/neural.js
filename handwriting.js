var network = null;
$(document).ready(function() {
    network = neuraljs.constructors.fullyConnected([28*28,14*14,26]);
    network.log = function() {};
    // assign each output to a letter
    var outputs = network.getOutputs();
    var letters = {};
    var j=0;
    var a = ("a".charCodeAt(0));
    for(var id in outputs) {
        letters[ String.fromCharCode( a + j ) ] = id;
        j++;
    }
    network.hi = 1;
    network.lo = -1;
    network.outputIdsByLetter = letters;
    console.log(network.outputIdsByLetter);
});

function getActualData() {
    var primaryCanvas = $('#canvas')[0];
    
    // normalize to our input format and attach to the new canvas for the training set
    var ctx = primaryCanvas.getContext('2d');
    var imgData = ctx.getImageData(0,0,primaryCanvas.width,primaryCanvas.height);
    var actualData = [];
    for (var i=0;i<imgData.data.length;i+=4)
    {
        if( imgData.data[i+0]==imgData.data[i+1]
            && imgData.data[i+1]==imgData.data[i+2]
            && imgData.data[i+2]==255 ) {
            actualData[i/4] = network.lo;
        } else actualData[i/4] = network.hi;
    }
    for(var i=0; i<primaryCanvas.height; i++) {
        var s = "";
        for(var j=0; j<primaryCanvas.width; j++) {
            s = s+','+actualData[(primaryCanvas.width*i)+j];
        }
        console.log(s);
    }
    return actualData;
}

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
$('#add').mouseup(function(evt) {
        
    var primaryCanvas = $('#canvas')[0];
    var actualData = getActualData();
    
    // duplicate the properties of our primary canvas
    var newSample = $('<div style="clear:both;"><canvas style="border:1px solid blue;"></canvas></div>')[0];
    var newSampleCanvas = $('canvas',newSample)[0];
    newSampleCanvas.width = primaryCanvas.width;
    newSampleCanvas.height = primaryCanvas.height;
    newSampleCanvas.actualData = actualData;
    
    // actually fill in the new canvas with data from our primary
    var newSampleCtx = newSampleCanvas.getContext('2d');
    newSampleCtx.createImageData(primaryCanvas.width,primaryCanvas.height);
    newSampleCtx.putImageData(imgData,0,0);
    
    var outputSelect = $('<select></select>')[0];
    for(var letter in network.outputIdsByLetter) {
        var newOpt = $('<option value="'+letter+'">'+letter+'</option>')[0];
        $(outputSelect).append(newOpt);
    }
    $(newSample).append(outputSelect);
    newSampleCanvas.select = outputSelect;
    
    // tack the sample on to the end
    $('#training_set').append(newSample);
    
    // reset the primary canvas
    $('#canvas').each(function() {
        var ctx = this.getContext("2d");
        ctx.fillStyle = 'white';
        ctx.fillRect(0,0,this.width,this.height);
    });
});
$('#train').click(function(evt) {
    var samples = [];
    $('canvas',$('#training_set')).each(function() {
        var sample = {}
        var i=0;
        for(var id in network.getInputs()) {
            sample[id] = this.actualData[i];
            i++;
        }
        for(var id in network.getOutputs()) {
            sample[id] = network.lo;
        }
        sample[network.outputIdsByLetter[$(this.select).val()]] = network.hi;
        console.log('letter = '+$(this.select).val());
        console.log(sample);
        samples.push(sample);
    });
    var sampler = $N.samplers.shuffler();
    sampler.samples = samples;
    network.train(sampler);
});
$('#evaluate').click(function(evt) {
    var actualData = getActualData();
    $.ajax('neural.php',{
        method:'POST',
        data:{
            data:actualData.join(','),
            lo:network.lo,
            hi:network.hi,
            op:'evaluate'
        },
        success:function(data) {
            console.log('success');
            console.log(data);
        }
    });
});
$('#load').click(function(evt) {
    console.log('on load');
    var actualData = getActualData();
    $.ajax('neural.php',{
        method:'POST',
        data:{
            db:'neuraljs',
            col:'handwriting',
            key:'handwriting',
            op:'select'
        },
        success:function(data) {
            console.log('success');
            console.log(data);
        }
    });
});

$('#save').click(function(evt) {
    network.save('handwriting');
});
