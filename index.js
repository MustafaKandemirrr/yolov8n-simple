const path = require('path');
const ort = require('onnxruntime-node');
const sharp = require('sharp');
const { performance } = require('perf_hooks');

const MODEL_PATH = path.join(__dirname, 'yolov8n.onnx');
const INPUT_IMAGE = path.join(__dirname, 'araba.jpg');
const CONF_THRESHOLD = 0.45;
const IOU_THRESHOLD = 0.50;
const TOTAL_CAPACITY = 20;

function iou(box1, box2) {
    const x1 = Math.max(box1[0], box2[0]);
    const y1 = Math.max(box1[1], box2[1]);
    const x2 = Math.min(box1[2], box2[2]);
    const y2 = Math.min(box1[3], box2[3]);
    
    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    const area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    
    return intersection / (area1 + area2 - intersection);
}

function nms(boxes) {
    if (boxes.length === 0) return [];
    boxes.sort((a, b) => b.prob - a.prob);
    const result = [];
    while (boxes.length > 0) {
        const best = boxes.shift();
        result.push(best);
        for (let i = boxes.length - 1; i >= 0; i--) {
            if (iou(best.box, boxes[i].box) > IOU_THRESHOLD) {
                boxes.splice(i, 1);
            }
        }
    }
    return result;
}

function bytesToMb(bytes) {
    return bytes / (1024 * 1024);
}

function memorySnapshot() {
    const m = process.memoryUsage();
    return {
        rss: bytesToMb(m.rss),
        heapUsed: bytesToMb(m.heapUsed),
        heapTotal: bytesToMb(m.heapTotal),
        external: bytesToMb(m.external)
    };
}

function printPerformance(stageTimes, startMem, endMem, totalMs) {
    console.log("\n============= PERFORMANS RAPORU =============");
    for (const stage of stageTimes) {
        console.log(`${stage.name.padEnd(20)}: ${stage.ms.toFixed(2)} ms`);
    }
    console.log(`Toplam süre         : ${totalMs.toFixed(2)} ms`);
    console.log("---------------------------------------------");
    console.log(`RSS                 : ${startMem.rss.toFixed(2)} -> ${endMem.rss.toFixed(2)} MB`);
    console.log(`Heap Used           : ${startMem.heapUsed.toFixed(2)} -> ${endMem.heapUsed.toFixed(2)} MB`);
    console.log(`Heap Total          : ${startMem.heapTotal.toFixed(2)} -> ${endMem.heapTotal.toFixed(2)} MB`);
    console.log(`External            : ${startMem.external.toFixed(2)} -> ${endMem.external.toFixed(2)} MB`);
    console.log("=============================================\n");
}

async function main() {
    try {
        const runStart = performance.now();
        const memStart = memorySnapshot();
        const stageTimes = [];

        let stageStart = performance.now();
        const session = await ort.InferenceSession.create(MODEL_PATH);
        stageTimes.push({ name: "Model yükleme", ms: performance.now() - stageStart });

        stageStart = performance.now();
        const { data } = await sharp(INPUT_IMAGE)
            .resize(320, 320, { fit: 'fill' })
            .removeAlpha()
            .raw()
            .toBuffer({ resolveWithObject: true });

        const float32Data = new Float32Array(3 * 320 * 320);
        const size = 320 * 320;
        for (let i = 0; i < size; i++) {
            float32Data[0 * size + i] = data[i * 3 + 0] / 255.0;
            float32Data[1 * size + i] = data[i * 3 + 1] / 255.0;
            float32Data[2 * size + i] = data[i * 3 + 2] / 255.0;
        }
        const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, 320, 320]);
        stageTimes.push({ name: "Ön işleme", ms: performance.now() - stageStart });

        stageStart = performance.now();
        const results = await session.run({ images: inputTensor });
        const output = results.output0.data;
        stageTimes.push({ name: "Model çıkarımı", ms: performance.now() - stageStart });

        stageStart = performance.now();
        const numAnchors = 2100;
        const predictions = [];

        for (let i = 0; i < numAnchors; i++) {
            const score = output[6 * numAnchors + i];
            if (score > CONF_THRESHOLD) {
                const cx = output[0 * numAnchors + i];
                const cy = output[1 * numAnchors + i];
                const w  = output[2 * numAnchors + i];
                const h  = output[3 * numAnchors + i];
                const x1 = cx - w / 2;
                const y1 = cy - h / 2;
                const x2 = cx + w / 2;
                const y2 = cy + h / 2;

                predictions.push({ box: [x1, y1, x2, y2], prob: score });
            }
        }
        stageTimes.push({ name: "Kutu çıkarımı", ms: performance.now() - stageStart });

        stageStart = performance.now();
        const finalCars = nms(predictions);
        const carCount = finalCars.length;
        const occupancy = (carCount / TOTAL_CAPACITY) * 100;
        stageTimes.push({ name: "NMS", ms: performance.now() - stageStart });
        const totalMs = performance.now() - runStart;
        const memEnd = memorySnapshot();
        
        console.clear();
        console.log("========================================");
        console.log(`📸 Analiz Tamamlandı`);
        console.log(`🚗 Tespit Edilen Araç: ${carCount}`);
        console.log(`🅿️  Doluluk Oranı:     %${occupancy.toFixed(1)}`);
        
        if (occupancy >= 100) console.log("🔴 DURUM: TAMAMEN DOLU");
        else if (occupancy > 80) console.log("🟠 DURUM: YOĞUN");
        else console.log("🟢 DURUM: MÜSAİT");
        console.log("========================================");
        printPerformance(stageTimes, memStart, memEnd, totalMs);

    } catch (error) {
        console.error("Hata:", error);
    }
}

main();