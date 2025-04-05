// app.js
const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

const app = express();

// multer: 메모리 저장 방식
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

app.use(express.json({ limit: '50mb' })); // JSON 데이터 처리

// 최종 결과 이미지 파일을 저장할 경로 (static 폴더)
const STATIC_DIR = path.join(__dirname, 'static');
const FINAL_DIR = path.join(STATIC_DIR, 'final');
if (!fs.existsSync(FINAL_DIR)) {
  fs.mkdirSync(FINAL_DIR, { recursive: true });
}

/**
 * AI 서버에 파일을 multipart/form-data 형식으로 전송하는 함수
 * @param {string} url - AI 서버 엔드포인트 (예: detect, highlight)
 * @param {Buffer} fileBuffer - 현재 이미지 버퍼
 * @param {string} mimeType - 파일의 mimetype (예: 'image/png')
 * @param {object} fields - 추가로 전송할 필드 (예: { object: "바나나", highlightMethod: "파란 테두리" })
 * @returns {Promise<object>} - AI 서버 JSON 응답 (예: { fileUrl: "...", detections: [...] })
 */
async function sendToAIServer(url, fileBuffer, mimeType, fields) {
  const form = new FormData();
  form.append('image', fileBuffer, { filename: 'upload.jpg', contentType: mimeType });
  // 추가 필드들을 form-data에 추가 (문자열 형식)
  for (const key in fields) {
    form.append(key, fields[key]);
  }
  const response = await axios.post(url, form, {
    headers: form.getHeaders(),
    responseType: 'json'
  });
  return response.data;
}

/**
 * 최종 이미지 버퍼를 파일로 저장하고 URL 반환
 * @param {Buffer} fileBuffer - 이미지 버퍼
 * @param {string} prefix - 파일 접두사
 * @returns {string} - 최종 이미지 파일 URL
 */
function saveFinalImage(fileBuffer, prefix = "final") {
  const filename = `${prefix}_${Date.now()}.jpg`;
  const filePath = path.join(FINAL_DIR, filename);
  fs.writeFileSync(filePath, fileBuffer);
  // 실제 운영 환경에 맞게 static URL 경로를 구성합니다.
  const fileUrl = `http://localhost:3000/static/final/${filename}`;
  return fileUrl;
}

/**
 * /process 엔드포인트: 클라이언트로부터 파일과 pipeline JSON을 받아 순차적으로 AI 서버 호출
 */
app.post('/process', upload.single('image'), async (req, res) => {
  try {
    let pipelineMetadata = [];
    let currentBuffer = null;
    let currentMimeType = null;

    // 파일 업로드 처리 (multipart/form-data)
    if (req.file) {
      currentBuffer = req.file.buffer;
      currentMimeType = req.file.mimetype;
      pipelineMetadata.push({ step: '입력', description: '이미지 파일 업로드 완료' });
    } else {
      return res.status(400).json({ error: '이미지 파일이 첨부되지 않았습니다.' });
    }

    // pipeline JSON 데이터 (문자열) 파싱
    let pipeline = [];
    if (req.body.pipeline) {
      try {
        pipeline = JSON.parse(req.body.pipeline);
      } catch (err) {
        return res.status(400).json({ error: 'pipeline 데이터가 올바른 JSON 형식이 아닙니다.' });
      }
    }

    // pipeline 내 각 단계 순차 처리
    for (const block of pipeline) {
      // 객체 감지 단계
      if (block.블록타입 === '로직' && block.로직타입 === '객체 감지') {
        const aiData = await sendToAIServer(
          'http://127.0.0.1:8000/detect',
          currentBuffer,
          currentMimeType,
          { object: block.객체 }
        );
        pipelineMetadata.push({
          step: '객체 감지',
          object: block.객체,
          detections: aiData.detections
        });
        // AI 서버가 반환한 fileUrl을 통해 파일을 다운로드하여 버퍼 업데이트
        const fileResp = await axios.get(aiData.fileUrl, { responseType: 'arraybuffer' });
        currentBuffer = Buffer.from(fileResp.data);
      }
      // 감지한 객체 강조 단계
      else if (block.블록타입 === '로직' && block.로직타입 === '감지한 객체 강조') {
        const aiData = await sendToAIServer(
          'http://127.0.0.1:8000/highlight',
          currentBuffer,
          currentMimeType,
          { object: block.객체, highlightMethod: block.강조방법 }
        );
        pipelineMetadata.push({
          step: '강조 처리',
          object: block.객체,
          method: block.강조방법
        });
        const fileResp = await axios.get(aiData.fileUrl, { responseType: 'arraybuffer' });
        currentBuffer = Buffer.from(fileResp.data);
      }
      // 출력 블록은 별도의 처리 없이 최종 결과 반환
    }

    // 최종 처리된 이미지 버퍼를 파일로 저장하고 URL 생성
    const finalImageUrl = saveFinalImage(currentBuffer, "final");

    res.json({
      processedImageUrl: finalImageUrl,
      pipeline: pipelineMetadata
    });
  } catch (error) {
    console.error("Processing error:", error.response ? error.response.data : error.message);
    res.status(500).json({ error: error.response ? error.response.data : error.message });
  }
});

// 정적 파일 제공: static 폴더 내 모든 파일 제공
app.use('/static', express.static(path.join(__dirname, 'static')));

app.listen(3000, () => console.log('API Server running on port 3000'));
