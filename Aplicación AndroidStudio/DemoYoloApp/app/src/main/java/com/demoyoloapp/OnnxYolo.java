package com.demoyoloapp;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.util.Log;
import android.graphics.RectF;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

public class OnnxYolo {

    private static final String TAG = "OnnxYolo";
    private final OrtEnvironment env;
    private final OrtSession session;
    private final int inputSize;
    private final float confThresh;
    private final float iouThresh = 0.45f;

    public static class Detection {
        public final int cls;
        public final float conf;
        public final float x1, y1, x2, y2;
        public Detection(int cls, float conf, float x1, float y1, float x2, float y2) {
            this.cls = cls;
            this.conf = conf;
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
        }
    }

    public OnnxYolo(OrtEnvironment env, OrtSession session, int inputSize, float confThresh) {
        this.env = env;
        this.session = session;
        this.inputSize = inputSize;
        this.confThresh = confThresh;
    }

    public List<Detection> infer(Bitmap bmp) {
        try {
            // 1) Pre-procesado → NL-CHW FloatBuffer
            final int N = 1, C = 3, H = inputSize, W = inputSize;
            ByteBuffer bb = ByteBuffer.allocateDirect(N * C * H * W * 4)
                    .order(ByteOrder.nativeOrder());
            FloatBuffer fb = bb.asFloatBuffer();
            int[] pixels = new int[H * W];
            bmp.getPixels(pixels, 0, W, 0, 0, W, H);
            for (int c = 0; c < C; c++) {
                for (int y = 0; y < H; y++) {
                    for (int x = 0; x < W; x++) {
                        int px = pixels[y * W + x];
                        float v = ( (c==0?Color.red(px): c==1?Color.green(px): Color.blue(px)) / 255.0f );
                        fb.put(v);
                    }
                }
            }
            fb.rewind();

            // 2) Inference
            OnnxTensor tensor = OnnxTensor.createTensor(env, fb, new long[]{N,C,H,W});
            Map<String, OnnxTensor> inputs = Collections.singletonMap(
                    session.getInputNames().iterator().next(), tensor
            );
            OrtSession.Result out = session.run(inputs);

            // 3) Raw output [1][C_out][M]
            float[][][] raw = (float[][][]) out.get(0).getValue();
            int C_out = raw[0].length;   // e.g. 5: [cx, cy, w, h, obj_conf]
            int M     = raw[0][0].length; // e.g. 8400

            // 4) Generar candidatas y decodificar
            List<RectF> boxes = new ArrayList<>();
            List<Float> scores = new ArrayList<>();
            for (int i = 0; i < M; i++) {
                float objConf = raw[0][4][i];
                if (objConf < confThresh) continue;
                float cx = raw[0][0][i];
                float cy = raw[0][1][i];
                float w  = raw[0][2][i];
                float h  = raw[0][3][i];
                // de normalizado (0-1) a píxeles
                float x1 = (cx - w/2f);
                float y1 = (cy - h/2f);
                float x2 = (cx + w/2f);
                float y2 = (cy + h/2f);
                boxes.add(new RectF(x1,y1,x2,y2));
                scores.add(objConf);
            }

            tensor.close();
            out.close();

            // 5) NMS
            List<Integer> keep = nms(boxes, scores, iouThresh);

            // 6) Construir lista final
            List<Detection> results = new ArrayList<>();
            for (int idx : keep) {
                RectF r = boxes.get(idx);
                float conf = scores.get(idx);
                results.add(new Detection(0, conf, r.left, r.top, r.right, r.bottom));
            }

            Log.d(TAG, "infer() → completado, dets=" + results.size());
            return results;

        } catch (OrtException e) {
            Log.e(TAG, "ONNX inference error", e);
            return Collections.emptyList();
        }
    }

    /** Supresión de no-máximos clásica (greedy) */
    private static List<Integer> nms(List<RectF> boxes, List<Float> scores, float iouThresh) {
        int n = boxes.size();
        List<Integer> idxs = new ArrayList<>();
        for (int i = 0; i < n; i++) idxs.add(i);
        // ordenar índices por score descendente
        Collections.sort(idxs, (i,j) -> Float.compare(scores.get(j), scores.get(i)));
        List<Integer> keep = new ArrayList<>();
        boolean[] suppressed = new boolean[n];
        for (int _i = 0; _i < n; _i++) {
            int i = idxs.get(_i);
            if (suppressed[i]) continue;
            keep.add(i);
            RectF bi = boxes.get(i);
            for (int _j = _i+1; _j < n; _j++) {
                int j = idxs.get(_j);
                if (suppressed[j]) continue;
                RectF bj = boxes.get(j);
                if (iou(bi, bj) > iouThresh) suppressed[j] = true;
            }
        }
        return keep;
    }

    /** Intersection-over-Union entre dos RectF */
    private static float iou(RectF a, RectF b) {
        float interLeft   = Math.max(a.left,   b.left);
        float interTop    = Math.max(a.top,    b.top);
        float interRight  = Math.min(a.right,  b.right);
        float interBottom = Math.min(a.bottom, b.bottom);
        float interW = Math.max(0, interRight - interLeft);
        float interH = Math.max(0, interBottom - interTop);
        float interArea = interW * interH;
        float areaA = (a.right - a.left)*(a.bottom - a.top);
        float areaB = (b.right - b.left)*(b.bottom - b.top);
        return interArea / (areaA + areaB - interArea + 1e-6f);
    }
}
