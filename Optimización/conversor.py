#!/usr/bin/env python3
"""
conversor.py

Pipeline para optimizar un modelo YOLOv8m (.pt) y convertir ONNX→SavedModel→TFLite.
"""

import os
import sys
import shutil

import onnx
from onnxsim import simplify
from ultralytics import YOLO
import tensorflow as tf
from onnx_tf.backend import prepare
import onnxruntime as ort

def main(pt_path: str = "prueba11.pt"):
    # 1) CARGAR modelo YOLOv8m
    print("1) Cargando modelo YOLOv8m...")
    model = YOLO(pt_path)

    # 2) EXPORTAR a ONNX con opset=11 (para Unsqueeze v11) y shapes fijas
    print("2) Exportando a ONNX (opset=11, estático)...")
    onnx_export = model.export(
        format="onnx",
        opset=11,
        dynamic=False
    )
    onnx_path = onnx_export[0] if isinstance(onnx_export, list) else onnx_export
    print(f"   → ONNX guardado en: {onnx_path}")

    # 3) SIMPLIFICAR ONNX
    print("3) Simplificando ONNX con onnx-simplifier...")
    model_onnx = onnx.load(onnx_path)
    model_simp, check = simplify(model_onnx)
    if not check:
        raise RuntimeError("Falló onnx-simplifier")
    opt_path = "prueba11_opt.onnx"
    onnx.save(model_simp, opt_path)
    print(f"   → ONNX simplificado guardado en: {opt_path}")

    # 4) ONNX → SavedModel (TensorFlow) via onnx-tf
    print("4) Convirtiendo ONNX→SavedModel con onnx-tf...")
    onnx_model = onnx.load(opt_path)
    tf_rep = prepare(onnx_model)
    saved_model_dir = "prueba11_saved_model"
    if os.path.isdir(saved_model_dir):
        shutil.rmtree(saved_model_dir)
    tf_rep.export_graph(saved_model_dir)
    print(f"   → SavedModel guardado en: {saved_model_dir}")

    # 5) SavedModel → TFLite
    print("5) Convirtiendo SavedModel→TFLite con tf.lite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = "prueba11.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"   → TFLite guardado en: {tflite_path}")

    # 6) (Opcional) Preparar ONNX Runtime Mobile
    print("6) Inicializando ONNX Runtime Mobile...")
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    _ = ort.InferenceSession(opt_path, sess_opts)
    print("   → Sesión ONNX Runtime lista")

    print("\n✅ Pipeline completado. Archivos generados:")
    print(f" • {onnx_path}")
    print(f" • {opt_path}")
    print(f" • {saved_model_dir}/")
    print(f" • {tflite_path}")

if __name__ == "__main__":
    pt_file = sys.argv[1] if len(sys.argv) > 1 else "prueba11.pt"
    if not os.path.isfile(pt_file):
        print(f"Error: No existe el archivo '{pt_file}'", file=sys.stderr)
        sys.exit(1)
    main(pt_file)
