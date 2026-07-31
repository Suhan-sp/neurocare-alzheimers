/* Multimodal Alzheimer's Prediction System - Interactive JavaScript */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Microphone Live Audio Recorder Setup
    let mediaRecorder = null;
    let audioChunks = [];
    const recordBtn = document.getElementById('recordBtn');
    const recStatus = document.getElementById('recStatus');
    const audioInputFile = document.getElementById('speechFile');
    const audioPreview = document.getElementById('audioPreview');

    if (recordBtn) {
        recordBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            if (!mediaRecorder || mediaRecorder.state === 'inactive') {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];

                    mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0) {
                            audioChunks.push(event.data);
                        }
                    };

                    mediaRecorder.onstop = () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        const audioUrl = URL.createObjectURL(audioBlob);
                        if (audioPreview) {
                            audioPreview.src = audioUrl;
                            audioPreview.style.display = 'block';
                        }
                        // Create file object for input
                        const file = new File([audioBlob], "recorded_speech.wav", { type: "audio/wav" });
                        const container = new DataTransfer();
                        container.items.add(file);
                        audioInputFile.files = container.files;
                        if (recStatus) recStatus.innerText = "Recording saved: recorded_speech.wav";
                    };

                    mediaRecorder.start();
                    recordBtn.classList.add('recording');
                    if (recStatus) recStatus.innerText = "Recording... Speak clearly into microphone";
                } catch (err) {
                    alert("Microphone access error: " + err.message);
                }
            } else if (mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                recordBtn.classList.remove('recording');
            }
        });
    }

    // 2. Drag & Drop File Zone Preview Handlers
    setupDropzone('eegDropzone', 'eegFile', 'eegFileInfo');
    setupDropzone('speechDropzone', 'speechFile', 'speechFileInfo');

    function setupDropzone(dropzoneId, inputId, infoId) {
        const zone = document.getElementById(dropzoneId);
        const input = document.getElementById(inputId);
        const info = document.getElementById(infoId);

        if (!zone || !input) return;

        zone.addEventListener('click', () => input.click());

        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.style.borderColor = 'var(--accent-blue)';
        });

        zone.addEventListener('dragleave', () => {
            zone.style.borderColor = 'var(--border-color)';
        });

        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.style.borderColor = 'var(--border-color)';
            if (e.dataTransfer.files.length) {
                input.files = e.dataTransfer.files;
                if (info) info.innerText = "Selected: " + input.files[0].name;
            }
        });

        input.addEventListener('change', () => {
            if (input.files.length && info) {
                info.innerText = "Selected: " + input.files[0].name;
            }
        });
    }
});
