import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:typed_data';

class TFLiteService {
  Interpreter? _interpreter;

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('model1.tflite');
      print('Model loaded successfully');
    } catch (e) {
      print('Failed to load model: $e');
    }
  }

  Future<List<double>?> runModel(Uint8List inputData) async {
    if (_interpreter == null) {
      print('Error: Model is not loaded.');
      return null;
    }

    try {
      // Adjust input/output shapes based on your model
      List<List<double>> output = List.generate(1, (_) => List.filled(5, 0));

      _interpreter!.run(inputData, output);

      return output[0]; // Return the first output row
    } catch (e) {
      print('Model inference failed: $e');
      return null;
    }
  }

  void close() {
    _interpreter?.close();
  }
}
