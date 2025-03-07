import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'dart:io';
import 'dart:typed_data';
import 'tflite_service.dart';
import 'package:flutter/services.dart'; // Import rootBundle

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Image Classifier',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
      ),
      home: const MyHomePage(title: 'Image Classifier Home'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  TFLiteService tfliteService = TFLiteService();
  File? _image;
  String? _classificationResult;

  @override
  void initState() {
    super.initState();
    _testAssetLoad(); // <-- Test if model asset exists before loading the model
    tfliteService.loadModel();
  }

    Future<void> _testAssetLoad() async {
    try {
      await rootBundle.load('assets/model1.tflite');
      print('✅ Model asset verified: File exists and can be read');
    } catch (e) {
      print('❌ Model asset error: $e');
    }
  }

  Future<void> _pickImage() async {
    final image = await ImagePicker().pickImage(source: ImageSource.gallery);

    if (image == null) return;

    setState(() {
      _image = File(image.path);
      _classificationResult = null;
    });

    // Convert image to Uint8List for TFLite model
    Uint8List inputBytes = await _processImage(File(image.path));
    
    final results = await tfliteService.runModel(inputBytes);
    setState(() {
      _classificationResult = results?.isNotEmpty == true ? results.toString() : "No classification found";
    });
  }

  // Convert image to Uint8List & resize for model input
Future<Uint8List> _processImage(File imageFile) async {
  Uint8List imageBytes = await imageFile.readAsBytes();
  img.Image? image = img.decodeImage(imageBytes);

  if (image == null) {
    throw Exception("Failed to decode image");
  }

  // Resize image to match model's expected input size (e.g., 224x224)
  img.Image resized = img.copyResize(image, width: 224, height: 224);

  // Convert image to Uint8List (removing 'format' as it's no longer needed)
  return Uint8List.fromList(img.encodeJpg(resized));
}


  @override
  void dispose() {
    tfliteService.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            _image != null
                ? Image.file(_image!)
                : const Text("No Image Selected"),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _pickImage,
              child: const Text("Select Image"),
            ),
            const SizedBox(height: 20),
            _classificationResult != null
                ? Text(
                    "Result: $_classificationResult",
                    style: Theme.of(context).textTheme.headlineSmall,
                  )
                : const Text("No result yet"),
          ],
        ),
      ),
    );
  }
}
