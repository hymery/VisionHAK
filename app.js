class YOLODetector {
    constructor() {
        this.session = null;
        this.inputSize = 320;
        this.classNames = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'cat', 'dog', 'bird'];
    }

    async loadModel() {
        try {
            this.session = await ort.InferenceSession.create('./models/yolov8n.onnx', { executionProviders: ['webgl'] });
            console.log('✅ Модель загружена');
            return true;
        } catch (error) {
            console.error('❌ Ошибка загрузки:', error);
            return false;
        }
    }

    async detect(videoElement) {
        if (!this.session) throw new Error('Модель не загружена');
        const inputTensor = await this.preprocess(videoElement);
        const outputs = await this.session.run({ images: inputTensor });
        const detections = this.postprocess(outputs);
        inputTensor.dispose();
        return detections;
    }

    async preprocess(videoElement) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = this.inputSize;
        canvas.height = this.inputSize;
        ctx.drawImage(videoElement, 0, 0, this.inputSize, this.inputSize);
        const imageData = ctx.getImageData(0, 0, this.inputSize, this.inputSize);
        const rgbData = new Float32Array(this.inputSize * this.inputSize * 3);
        for (let i = 0; i < imageData.data.length; i += 4) {
            const pixelIndex = i / 4;
            rgbData[pixelIndex] = imageData.data[i] / 255.0;
            rgbData[pixelIndex + this.inputSize * this.inputSize] = imageData.data[i + 1] / 255.0;
            rgbData[pixelIndex + 2 * this.inputSize * this.inputSize] = imageData.data[i + 2] / 255.0;
        }
        return new ort.Tensor('float32', rgbData, [1, 3, this.inputSize, this.inputSize]);
    }

    postprocess(outputs) {
        const detections = [];
        const output = outputs.output0.data;
        for (let i = 0; i < 8400; i++) {
            const confidence = output[4 * 8400 + i];
            if (confidence > 0.3) {
                let maxClassProb = 0, classId = -1;
                for (let j = 0; j < 80; j++) {
                    const prob = output[(5 + j) * 8400 + i];
                    if (prob > maxClassProb) { maxClassProb = prob; classId = j; }
                }
                const finalConfidence = confidence * maxClassProb;
                if (finalConfidence > 0.4 && classId !== -1) {
                    const className = this.classNames[classId];
                    if (className) {
                        detections.push({
                            class: className,
                            confidence: finalConfidence,
                            bbox: [output[i], output[8400 + i], output[2 * 8400 + i], output[3 * 8400 + i]]
                        });
                    }
                }
            }
        }
        return detections;
    }

    estimateDistance(bbox) {
        const area = bbox[2] * bbox[3];
        if (area > 0.2) return { level: 'close', text: 'близко' };
        if (area > 0.05) return { level: 'medium', text: 'средняя дистанция' };
        return { level: 'far', text: 'далеко' };
    }
}

class NavigationAssistant {
    constructor() {
        this.detector = new YOLODetector();
        this.isRunning = false;
        this.videoElement = document.getElementById('webcam');
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.statusElement = document.getElementById('status');
        this.objectsElement = document.getElementById('objects');
        this.bindEvents();
    }

    bindEvents() {
        this.startBtn.addEventListener('click', () => this.start());
        this.stopBtn.addEventListener('click', () => this.stop());
    }

    async init() {
        try {
            this.updateStatus('🔄 Загрузка модели...');
            await this.detector.loadModel();
            await this.initCamera();
            this.updateStatus('✅ Готов к работе');
        } catch (error) {
            this.updateStatus('❌ Ошибка: ' + error.message);
        }
    }

    async initCamera() {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        this.videoElement.srcObject = stream;
    }

    async start() {
        this.isRunning = true;
        this.startBtn.disabled = true;
        this.stopBtn.disabled = false;
        this.updateStatus('🔍 Сканирование...');
        this.speak('Навигация активирована');
        this.detectionLoop();
    }

    stop() {
        this.isRunning = false;
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.updateStatus('⏹️ Остановлено');
        this.speak('Навигация остановлена');
    }

    async detectionLoop() {
        if (!this.isRunning) return;
        try {
            const detections = await this.detector.detect(this.videoElement);
            this.processDetections(detections);
        } catch (error) {
            console.error('Ошибка:', error);
        }
        setTimeout(() => this.detectionLoop(), 3000);
    }

    processDetections(detections) {
        const objects = {};
        detections.forEach(det => {
            if (!objects[det.class]) objects[det.class] = 0;
            objects[det.class]++;
        });

        let html = '';
        Object.entries(objects).forEach(([className, count]) => {
            html += `<div class="object-item"><span>${className}</span><span>${count} шт</span></div>`;
        });
        this.objectsElement.innerHTML = html || '<div class="object-item">Объекты не обнаружены</div>';

        if (detections.length > 0) {
            const text = `Обнаружено ${detections.length} объектов`;
            this.speak(text);
            this.updateStatus(text);
        }
    }

    speak(text) {
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'ru-RU';
            utterance.rate = 0.9;
            speechSynthesis.speak(utterance);
        }
    }

    updateStatus(message) {
        this.statusElement.textContent = message;
    }
}

const app = new NavigationAssistant();
window.addEventListener('load', () => app.init());