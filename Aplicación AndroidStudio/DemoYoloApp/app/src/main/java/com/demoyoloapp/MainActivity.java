package com.demoyoloapp;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.widget.Button;
import android.widget.ToggleButton;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private static final int PERM_CODE = 1001;
    private static final Size ANALYSIS_RESOLUTION = new Size(640, 640);

    private PreviewView previewView;
    private Button btnOn, btnOff, btnTest;
    private ToggleButton toggleModo;
    private MediaPlayer mediaPlayer;
    private OnnxYolo onnx;
    private ExecutorService cameraExecutor;
    private ProcessCameraProvider cameraProvider;

    // Control para saber si estamos en modo distancia (true) o modo objeto (false)
    private boolean esModoDistancia = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView    = findViewById(R.id.cameraPreview);
        btnOn          = findViewById(R.id.btnEncender);
        btnOff         = findViewById(R.id.btnApagar);
        btnTest        = findViewById(R.id.btnTest);
        toggleModo     = findViewById(R.id.toggleModo);
        btnOff.setVisibility(View.GONE);

        mediaPlayer    = MediaPlayer.create(this, R.raw.beep);
        cameraExecutor = Executors.newSingleThreadExecutor();

        // Listener para cambiar entre modo objeto y modo distancia
        toggleModo.setOnCheckedChangeListener((buttonView, isChecked) -> {
            esModoDistancia = isChecked;
            if (esModoDistancia) {
                Log.d(TAG, "Modo cambiado: MODO DISTANCIA");
            } else {
                Log.d(TAG, "Modo cambiado: MODO OBJETO");
            }
        });

        try {
            // 1) Copiar modelo ONNX desde assets a storage de la app
            String modelPath = assetFilePath("model.onnx");

            // 2) Crear OrtEnvironment y OrtSession
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);
            OrtSession session = env.createSession(modelPath, opts);

            // 3) Instanciar OnnxYolo con confThresh = 0.5f (o el que prefieras)
            onnx = new OnnxYolo(env, session, /*inputSize=*/640, /*confThresh=*/0.7f);
            Log.d(TAG, "OnnxYolo listo");

        } catch (Exception e) {
            Log.e(TAG, "Error cargando modelo", e);
            btnOn.setEnabled(false);
            btnTest.setEnabled(false);
            toggleModo.setEnabled(false);
        }

        // BOTÓN TEST: inferencia sobre imagen de prueba
        btnTest.setOnClickListener(v -> cameraExecutor.execute(() -> {
            Bitmap bmp = loadTestImage();
            int cnt = onnx.infer(bmp).size();
            Log.d(TAG, "Test detections=" + cnt);
            if (cnt > 0) runOnUiThread(this::playBeep);
        }));

        btnOn.setOnClickListener(v -> {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(
                        this,
                        new String[]{Manifest.permission.CAMERA},
                        PERM_CODE
                );
            } else {
                startCameraAnalysis();
                btnOn.setVisibility(View.GONE);
                btnOff.setVisibility(View.VISIBLE);
            }
        });

        btnOff.setOnClickListener(v -> {
            if (cameraProvider != null) cameraProvider.unbindAll();
            btnOff.setVisibility(View.GONE);
            btnOn.setVisibility(View.VISIBLE);
        });
    }

    private String assetFilePath(String name) throws Exception {
        File f = new File(getFilesDir(), name);
        byte[] buf = new byte[4096];
        if (!f.exists()) {
            try (InputStream is = getAssets().open(name);
                 FileOutputStream os = new FileOutputStream(f)) {
                int r;
                while ((r = is.read(buf)) != -1) {
                    os.write(buf, 0, r);
                }
                os.flush();
            }
        }
        return f.getAbsolutePath();
    }

    private Bitmap loadTestImage() {
        try (InputStream is = getAssets().open("test_image.jpg")) {
            Bitmap b = BitmapFactory.decodeStream(is);
            return Bitmap.createScaledBitmap(b, 640, 640, true);
        } catch (Exception e) {
            Log.e(TAG, "Error cargando test image", e);
            return Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888);
        }
    }

    private void startCameraAnalysis() {
        ListenableFuture<ProcessCameraProvider> fut =
                ProcessCameraProvider.getInstance(this);
        fut.addListener(() -> {
            try {
                cameraProvider = fut.get();

                Preview preview = new Preview.Builder()
                        .setTargetRotation(previewView.getDisplay().getRotation())
                        .build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setTargetResolution(ANALYSIS_RESOLUTION)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build();

                analysis.setAnalyzer(cameraExecutor, this::analyzeImage);

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(
                        this,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        analysis
                );
                Log.d(TAG, "Camera + Analysis bound; streaming inference activa");

            } catch (Exception e) {
                Log.e(TAG, "Error arrancando cameraAnalysis", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void analyzeImage(ImageProxy image) {
        int w = image.getWidth(), h = image.getHeight();
        int rotation = image.getImageInfo().getRotationDegrees();
        Log.d(TAG, "ENTER analyzeImage: size=" + w + "x" + h + " rot=" + rotation);

        Bitmap bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        ByteBuffer buf = image.getPlanes()[0].getBuffer();
        buf.rewind();
        bmp.copyPixelsFromBuffer(buf);

        Bitmap bmpToInfer = rotation != 0
                ? Bitmap.createBitmap(bmp, 0, 0, w, h, new Matrix() {{ postRotate(rotation); }}, true)
                : bmp;
        image.close();

        cameraExecutor.execute(() -> {
            Bitmap scaled = Bitmap.createScaledBitmap(bmpToInfer, 640, 640, true);
            Log.d(TAG, "About to infer on " + scaled.getWidth() + "x" + scaled.getHeight());
            final java.util.List<OnnxYolo.Detection> detections = onnx.infer(scaled);
            Log.d(TAG, "Stream detections=" + detections.size());

            if (detections.isEmpty()) {
                return;
            }

            if (!esModoDistancia) {
                // ─────────────── MODO OBJETO ───────────────
                runOnUiThread(this::playBeep);

            } else {
                // ─────────────── MODO DISTANCIA ───────────────
                // Toda la lógica de distancia la movemos al hilo UI
                runOnUiThread(() -> {
                    // 1) Seleccionamos la detección de mayor área en 640×640
                    OnnxYolo.Detection mejorDeteccion = seleccionarMejorPorArea(detections);

                    float anchoBBox640 = mejorDeteccion.x2 - mejorDeteccion.x1;
                    float altoBBox640  = mejorDeteccion.y2 - mejorDeteccion.y1;

                    // 2) Calculamos área de la bounding box y área total (640×640)
                    float areaBBox = anchoBBox640 * altoBBox640;
                    float areaFull = 640f * 640f;

                    // 3) Porcentaje de área que ocupa el objeto
                    float porcentajeArea = (areaBBox / areaFull) * 100f;
                    if (porcentajeArea > 100f) porcentajeArea = 100f;  // clamped

                    // 4) Número de pips
                    int numPips = (int)porcentajeArea;
                    if (numPips < 10)  numPips = 1;

                    Log.d(TAG, String.format(
                            "Bounds 640×640: [w=%.1f, h=%.1f] → área bbox=%.1f → %%Area=%.1f → pips=%d",
                            anchoBBox640, altoBBox640, areaBBox, porcentajeArea, numPips
                    ));

                    // 5) Reproducir numPips pips distribuidos en 1 segundo
                    playPipsEnUnSegundo(numPips);
                });
            }
        });
    }

    /** Selecciona la detección de mayor área (width * height) en coordenadas 0–640 */
    private OnnxYolo.Detection seleccionarMejorPorArea(java.util.List<OnnxYolo.Detection> detections) {
        OnnxYolo.Detection mejor = detections.get(0);
        float areaMax = (mejor.x2 - mejor.x1) * (mejor.y2 - mejor.y1);
        for (int i = 1; i < detections.size(); i++) {
            OnnxYolo.Detection d = detections.get(i);
            float area = (d.x2 - d.x1) * (d.y2 - d.y1);
            if (area > areaMax) {
                areaMax = area;
                mejor = d;
            }
        }
        return mejor;
    }

    /** Reproduce “count” pips distribuidos uniformemente en 1 segundo */
    private void playPipsEnUnSegundo(int count) {
        if (count <= 0) return;
        int num = 1;
        if(count>=50 && count<=65){num = 2;}
        else if(count>65 && count<=85){num = 3;}
        else if(count>85){num = 4;}
        final long intervaloMs = 1000L / num;
        Handler handler = new Handler(Looper.getMainLooper());

        for (int i = 0; i < num; i++) {
            long delay = i * intervaloMs;
            handler.postDelayed(this::playBeep, delay);
        }
    }

    private void playBeep() {
        if (mediaPlayer == null) return;
        if (mediaPlayer.isPlaying()) {
            mediaPlayer.seekTo(0);
        }
        mediaPlayer.start();
    }

    @Override
    public void onRequestPermissionsResult(
            int code, @NonNull String[] perms, @NonNull int[] res) {
        super.onRequestPermissionsResult(code, perms, res);
        if (code == PERM_CODE && res.length > 0 && res[0] == PackageManager.PERMISSION_GRANTED) {
            btnOn.performClick();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
        if (mediaPlayer != null) {
            mediaPlayer.release();
            mediaPlayer = null;
        }
    }
}
