import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'PapayaMaturityInfoScreen.dart';

class MaturityScreen extends StatefulWidget {
  @override
  _MaturityScreenState createState() => _MaturityScreenState();
}

class _MaturityScreenState extends State<MaturityScreen> {
  Uint8List? _imageBytes;
  String? _result;
  final ImagePicker _picker = ImagePicker();

  // Function to get ripeness color based on the prediction
  Color _getRipenessColor(String ripeness) {
    switch (ripeness.toLowerCase()) {
      case "not_mature":
        return Colors.teal;
      case "partially_mature":
        return Colors.lightGreen;
      case "mature":
        return Colors.green;
      case "rotten":
        return Colors.yellowAccent;
      default:
        return Colors.grey;
    }
  }

  Future<void> _uploadImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);

    if (image != null) {
      Uint8List imageBytes = await image.readAsBytes();
      setState(() {
        _imageBytes = imageBytes;
        _result = null; // Reset result when a new image is picked
      });

      var request = http.MultipartRequest(
        'POST',
        Uri.parse(
          'http://127.0.0.1:5000/predict',
        ), // Replace with your backend URL
      );

      request.files.add(
        http.MultipartFile.fromBytes('file', imageBytes, filename: 'image.jpg'),
      );

      var response = await request.send();
      var responseData = await response.stream.bytesToString();

      setState(() {
        _result = jsonDecode(responseData)['predicted_class'];
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          "Papaya Maturity",
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
        leading: Padding(
          padding: EdgeInsets.all(8.0),
          child: Image.asset("assets/Papaya.png"), // Add an icon for branding
        ),
      ),
      body: Padding(
        padding: EdgeInsets.all(8.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (_imageBytes != null)
              Column(
                children: [
                  Image.memory(_imageBytes!, height: 150),
                  SizedBox(height: 10),
                ],
              ),
            ElevatedButton(
              onPressed: _uploadImage,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.lightGreen[200], // Light green button
                padding: EdgeInsets.symmetric(horizontal: 20, vertical: 10),
              ),
              child: Text(
                "+ Your Image",
                style: TextStyle(fontSize: 16, color: Colors.black),
              ),
            ),
            SizedBox(height: 20),
            Text(
              "Papaya Ripeness Level is:",
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 10),
            if (_result != null)
              Container(
                padding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  color: _getRipenessColor(_result!),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  _result!,
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
              ),
            SizedBox(height: 30),
            // Ripeness Levels Legend
            Column(
              children: [
                Text(
                  "Ripeness Levels",
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 10),
                _buildLegend("Not Ripe", Colors.teal),
                _buildLegend("Partially Ripe", Colors.lightGreen),
                _buildLegend("Ripe", Colors.green),
                _buildLegend("Rotten", Colors.yellowAccent),
                SizedBox(height: 20),
                ElevatedButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => PapayaMaturityInfoScreen(),
                      ),
                    );
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.orange,
                  ),
                  child: Text(
                    "Learn About Maturity Levels",
                    style: TextStyle(color: Colors.white),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLegend(String label, Color color) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(width: 20, height: 10, color: color),
          SizedBox(width: 10),
          Text(label, style: TextStyle(fontSize: 14)),
        ],
      ),
    );
  }
}
