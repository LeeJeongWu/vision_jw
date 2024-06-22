function handleCheckbox(checkboxId, scriptName) {
    var checkbox = document.getElementById(checkboxId);
    if (checkbox.checked) {
        $.ajax({
            url: '/run_script',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ script_name: scriptName }),
            success: function(response) {
                console.log('Script output:', response.output);
                document.getElementById('output').innerText = response.output; // 결과를 HTML 요소에 삽입
            },
            error: function(error) {
                console.log('Error:', error);
                document.getElementById('output').innerText = 'Error executing script';
            }
        });
    }
}

function handleVideoCheckbox() {
    var videoCheckbox = document.getElementById('videoCheckbox');
    var videoInput = document.getElementById('videoInput');
    if (videoCheckbox.checked) {
        videoInput.style.display = 'block';
    } else {
        videoInput.style.display = 'none';
    }
}

function handleWebcamCheckbox() {
    var webcamCheckbox = document.getElementById('webcamCheckbox');
    var webcamInput = document.getElementById('webcamInput');
    if (webcamCheckbox.checked) {
        webcamInput.style.display = 'block';
    } else {
        webcamInput.style.display = 'none';
        stopWebcam();
    }
}

let webcamStream;
let captureInterval;

function startWebcam() {
    const video = document.getElementById('webcam');
    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            webcamStream = stream;
            video.srcObject = stream;
            captureInterval = setInterval(captureFrames, 1000); // 매 초마다 프레임 캡처
        })
        .catch((err) => {
            console.error('Error accessing webcam: ', err);
        });
}

function stopWebcam() {
    if (webcamStream) {
        const tracks = webcamStream.getTracks();
        tracks.forEach(track => track.stop());
        webcamStream = null;
    }
    if (captureInterval) {
        clearInterval(captureInterval);
    }
}

function captureFrames() {
    const video = document.getElementById('webcam');
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const frame = canvas.toDataURL('image/png');

    analyzeFrames([frame]);
}

function analyzeFrames(frames) {
    $.ajax({
        url: '/analyze',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ frames: frames }),
        success: function(response) {
            console.log('Analyze output:', response.response);
            document.getElementById('output').innerText = response.response + '\n' + response.response_to_response; // 결과를 HTML 요소에 삽입
            document.getElementById('speechBubble').innerText = response.response_to_response; // 말풍선에 텍스트 표시
        },
        error: function(error) {
            console.log('Error:', error);
            document.getElementById('output').innerText = 'Error analyzing webcam frames';
        }
    });
}

function uploadVideo() {
    var fileInput = document.getElementById('videoFile');
    var file = fileInput.files[0];
    if (!file) {
        alert('Please select a video file.');
        return;
    }

    var formData = new FormData();
    formData.append('video', file);

    $.ajax({
        url: '/upload_video',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            console.log('Video upload output:', response.output);
            document.getElementById('output').innerText = response.output; // 결과를 HTML 요소에 삽입
        },
        error: function(error) {
            console.log('Error:', error);
            document.getElementById('output').innerText = 'Error uploading video';
        }
    });
}