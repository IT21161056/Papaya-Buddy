import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';

class AskCommunityView extends StatefulWidget {
  @override
  _AskCommunityScreenState createState() => _AskCommunityScreenState();
}

class _AskCommunityScreenState extends State<AskCommunityView> {
  TextEditingController questionController = TextEditingController();
  TextEditingController descriptionController = TextEditingController();
  File? _selectedImage;

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      setState(() {
        _selectedImage = File(pickedFile.path);
      });
    }
  }

  void _removeImage() {
    setState(() {
      _selectedImage = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 1,
        leading: IconButton(
          icon: Icon(Icons.arrow_back, color: Colors.black),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text(
          "Ask Community",
          style: TextStyle(color: Colors.black),
        ),
      ),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (_selectedImage != null)
                Column(
                  children: [
                    Stack(
                      children: [
                        Container(
                          height: 200,
                          width: double.infinity,
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(8),
                            image: DecorationImage(
                              image: FileImage(_selectedImage!),
                              fit: BoxFit.cover,
                            ),
                          ),
                        ),
                        Positioned(
                          top: 8,
                          right: 8,
                          child: IconButton(
                            icon: Icon(Icons.close, color: Colors.white),
                            onPressed: _removeImage,
                            style: IconButton.styleFrom(
                              backgroundColor: Colors.black54,
                            ),
                          ),
                        ),
                      ],
                    ),
                    SizedBox(height: 16),
                  ],
                ),
              ElevatedButton.icon(
                onPressed: _pickImage,
                icon: Icon(Icons.add_photo_alternate),
                label: Text(_selectedImage == null ? "Add Image" : "Change Image"),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.white,
                  foregroundColor: Colors.black,
                  side: BorderSide(color: Colors.grey.shade300),
                ),
              ),
              SizedBox(height: 16),
             
              Text("Your question to the community",
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
              SizedBox(height: 8),
              TextField(
                controller: questionController,
                maxLength: 200,
                maxLines: 3,
                decoration: InputDecoration(
                  hintText:
                      "Add a question indicating what's wrong with your crop",
                  border: OutlineInputBorder(),
                ),
              ),
              SizedBox(height: 24),
              Text("Description of your problem",
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
              SizedBox(height: 8),
              TextField(
                controller: descriptionController,
                maxLength: 2500,
                maxLines: 6,
                decoration: InputDecoration(
                  hintText:
                      "Describe specialities such as change of leaves, root colour, bugs, tears...",
                  border: OutlineInputBorder(),
                ),
              ),
            ],
          ),
        ),
      ),
      bottomNavigationBar: Padding(
        padding: EdgeInsets.all(16.0),
        child: ElevatedButton(
          onPressed: () {},
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.blue,
            foregroundColor: Colors.white,
            padding: EdgeInsets.symmetric(vertical: 16),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(30),
            ),
          ),
          child: Text("Send", style: TextStyle(fontSize: 16)),
        ),
      ),
    );
  }
}