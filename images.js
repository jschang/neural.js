var network = $N.constructors.fullyConnected([28*28,28*28]);
for(var id in network.neurons) {
    network.neurons[id].activator = $N.activators.sigmoid;
}
network.lo = 0;
network.hi = 1;
network.log = function() {};
network.lastRun = null;

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

function isClose(a,b,epi) {
    if(typeof(epi)=='undefined') {
        epi = .00001;
    }
    return Math.abs(a-b) < epi;
}

var canvasHandler = {
    canvas:null,
    image:null,
    'mousemove':function(evt) {
        var ctx = canvasHandler.canvas.getContext('2d');
        var coords = canvasCoords(evt);
        ctx.clearRect(0,0,this.width,this.height);
        ctx.drawImage(canvasHandler.image,0,0);
        ctx.strokeStyle = "black 1px";
        ctx.strokeRect(coords.x, coords.y, 28, 28);
    },
    'mouseup':function(evt) {
        var ctx = canvasHandler.canvas.getContext('2d');
        var coords = canvasCoords(evt);
        var data = ctx.getImageData(coords.x,coords.y,28,28).data;
        var inputs = network.getInputs();
        var sample = {};
        var idx = 0;
        for(var id in inputs) {
            var lum = (0.299*data[idx] + 0.587*data[idx+1] + 0.114*data[idx+2])/255.0;
            sample[id] = lum;
            idx+=4;
        }
        var runData = network.forward(sample);
        console.log('runData:');
        console.log(runData);
        console.log('sample:');
        console.log(sample);
        var outputs = network.getOutputs();
        var out = {};
        for(var id in outputs) {
            out[id] = runData.activations[id];
        }
        console.log('output:');
        console.log(out);
        
        if(network.lastRun!=null) {
            var simAct = {};
            console.log('similar activations:');
            for(var id in network.neurons) {
                if( isClose(network.lastRun.activations[id],runData.activations[id]) && runData.activations[id]!=1) {
                    simAct[id] = {
                        last:network.lastRun.activations[id],
                        cur:runData.activations[id],
                        delta:Math.abs(network.lastRun.activations[id]-runData.activations[id])
                    };
                }
            }
            console.log(simAct);
        }
        network.lastRun = runData;
    },
    'mousedown':function(evt) {
    }
};

$('#load_url').click(function(evt) {
    var url = $('#url').val();
    var image = $('<img/>')[0];
    $(image).load(function(evt) {
        var canvas = $('<canvas/>')[0];
        canvas.style.height = image.naturalHeight+'px';
        canvas.style.width = image.naturalWidth+'px';
        canvas.height = image.naturalHeight;
        canvas.width = image.naturalWidth;
        canvas.style.border = "1px solid black";
        var ctx = canvas.getContext('2d');
        ctx.drawImage(image,0,0);
        $('#canvas_container').empty();
        $('#canvas_container').append(canvas);
        canvasHandler.canvas = canvas;
        canvasHandler.image = image;
        for(var handlerName in {mousemove:null,mouseup:null,mousedown:null}) {
            $(canvas)[handlerName](canvasHandler[handlerName]);
        }
    });
    image.src = url;
});
$('#url').attr('value','turkey-tacos.png');
$('#load_url').click();


