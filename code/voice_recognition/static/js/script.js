
// record button behavior when clicked
$('#recButton').addClass("notRec");
$('#recButton').click(function () {
    if ($('#recButton').hasClass('notRec')) {
        $('#recButton').removeClass("notRec");
        $('#recButton').addClass("Rec");


    }

});

//  initialize wavesurfer to visualize audio
var wavesurfer = WaveSurfer.create({
    container: '#waveform',
    waveColor: 'red',
    progressColor: 'black',
    hideScrollbar: true,
});


// function to start record
var startrecordingbutton = document.getElementById("recButton")
var playButton = document.getElementById("playButton");

var leftchannel = [];
var rightchannel = [];
var recorder = null;
var recordingLength = 0;
var volume = null;
var mediaStream = null;
var sampleRate = 48000;
var context = null;
var blob = null;

startrecordingbutton.addEventListener("click", function () {
    document.getElementById("cont1").remove()
    document.getElementById("cont2").remove()
    document.getElementById("cont3").remove()
    document.getElementById("waveform").hidden = true
    // Initialize recorder
    navigator.getUserMedia =
        navigator.getUserMedia ||
        navigator.webkitGetUserMedia ||
        navigator.mozGetUserMedia ||
        navigator.msGetUserMedia;
    navigator.getUserMedia(
        {
            audio: true,
        },
        function (e) {
            console.log("user consent");

            // creates the audio context
            window.AudioContext =
                window.AudioContext || window.webkitAudioContext;
            context = new AudioContext();

            // creates an audio node from the microphone incoming stream
            mediaStream = context.createMediaStreamSource(e);

            // https://developer.mozilla.org/en-US/docs/Web/API/AudioContext/createScriptProcessor
            // bufferSize: the onaudioprocess event is called when the buffer is full
            var bufferSize = 2048;
            var numberOfInputChannels = 2;
            var numberOfOutputChannels = 2;
            if (context.createScriptProcessor) {
                recorder = context.createScriptProcessor(
                    bufferSize,
                    numberOfInputChannels,
                    numberOfOutputChannels
                );
            } else {
                recorder = context.createJavaScriptNode(
                    bufferSize,
                    numberOfInputChannels,
                    numberOfOutputChannels
                );
            }

            recorder.onaudioprocess = function (e) {
                leftchannel.push(
                    new Float32Array(e.inputBuffer.getChannelData(0))
                );
                rightchannel.push(
                    new Float32Array(e.inputBuffer.getChannelData(1))
                );
                recordingLength += bufferSize;
            };

            // we connect the recorder
            mediaStream.connect(recorder);
            recorder.connect(context.destination);
        },
        function (e) {
            console.error(e);
        }
    );

    setTimeout(stopRecording, "3000");

});

function stopRecording() {
    $('#recButton').removeClass("Rec");
    $('#recButton').addClass("notRec");
    // stop recording
    recorder.disconnect(context.destination);
    mediaStream.disconnect(recorder);

    // we flat the left and right channels down
    // Float32Array[] => Float32Array
    var leftBuffer = flattenArray(leftchannel, recordingLength);
    var rightBuffer = flattenArray(rightchannel, recordingLength);
    // we interleave both channels together
    // [left[0],right[0],left[1],right[1],...]
    var interleaved = interleave(leftBuffer, rightBuffer);

    // we create our wav file
    var buffer = new ArrayBuffer(44 + interleaved.length * 2);
    var view = new DataView(buffer);

    // RIFF chunk descriptor
    writeUTFBytes(view, 0, "RIFF");
    view.setUint32(4, 44 + interleaved.length * 2, true);
    writeUTFBytes(view, 8, "WAVE");
    // FMT sub-chunk
    writeUTFBytes(view, 12, "fmt ");
    view.setUint32(16, 16, true); // chunkSize
    view.setUint16(20, 1, true); // wFormatTag
    view.setUint16(22, 2, true); // wChannels: stereo (2 channels)
    view.setUint32(24, sampleRate, true); // dwSamplesPerSec
    view.setUint32(28, sampleRate * 4, true); // dwAvgBytesPerSec
    view.setUint16(32, 4, true); // wBlockAlign
    view.setUint16(34, 16, true); // wBitsPerSample
    // data sub-chunk
    writeUTFBytes(view, 36, "data");
    view.setUint32(40, interleaved.length * 2, true);

    // write the PCM samples
    var index = 44;
    var volume = 1;
    for (var i = 0; i < interleaved.length; i++) {
        view.setInt16(index, interleaved[i] * (0x7fff * volume), true);
        index += 2;
    }

    // our final blob
    blob = new Blob([view], { type: "audio/wav" });

    saveRecord(blob);

    leftchannel = [];
    rightchannel = [];
    recordingLength = 0;
}


function flattenArray(channelBuffer, recordingLength) {
    var result = new Float32Array(recordingLength);
    var offset = 0;
    for (var i = 0; i < channelBuffer.length; i++) {
        var buffer = channelBuffer[i];
        result.set(buffer, offset);
        offset += buffer.length;
    }
    return result;
}

function interleave(leftChannel, rightChannel) {
    var length = leftChannel.length + rightChannel.length;
    var result = new Float32Array(length);

    var inputIndex = 0;

    for (var index = 0; index < length;) {
        result[index++] = leftChannel[inputIndex];
        result[index++] = rightChannel[inputIndex];
        inputIndex++;
    }
    return result;
}

function writeUTFBytes(view, offset, string) {
    for (var i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

// ajax to send requests and responce responces
let saveRecord = (audioBlob) => {
    let formdata = new FormData();
    formdata.append("AudioFile", audioBlob, "recordedAudio.wav");
    $.ajax({
        type: "POST",
        url: "http://127.0.0.1:5000/saveRecord",
        data: formdata,
        contentType: false,
        cache: false,
        processData: false,
        success: function (res) {
            var responce = JSON.parse(res)

            var big_cont = document.getElementById("images_container")
            var small_cont = document.getElementById("small-container")
            document.getElementById("waveform").hidden = false



            var element = document.createElement("div")
            element.className = "container"
            element.id = "cont1"
            element.innerHTML = responce[0]
            small_cont.appendChild(element)


            var img_bar = document.createElement("div")
            img_bar.className = "container"
            img_bar.id = "cont2"
            img_bar.innerHTML = responce[1]
            big_cont.appendChild(img_bar)

            var img_spect = document.createElement("div")
            img_spect.className = "container"
            img_spect.id = "cont3"
            img_spect.innerHTML = responce[2]
            big_cont.appendChild(img_spect)
            wavesurfer.load("./static/assets/recordedAudio.wav")



        },
    });
};



