import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:mobile_app/services/auth_services.dart';
import 'package:mobile_app/utils/constants.dart';
import 'dart:io';
import 'dart:convert';

class AskCommunityView extends StatefulWidget {
  @override
  _AskCommunityScreenState createState() => _AskCommunityScreenState();
}

class _AskCommunityScreenState extends State<AskCommunityView> {
  final TextEditingController questionController = TextEditingController();
  final TextEditingController descriptionController = TextEditingController();
  final AuthService _authService = AuthService();
  File? _selectedImage;
  bool _isLoading = false;

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

  Future<Map<String, dynamic>> createPost({
    required String userId,
    required String name,
    required String question,
    required String description,
    File? uploadedImg,
  }) async {
    final String url = '${BaseURL.BASE_URL}:5080/api/v1/community';
    
    try {
      final request = http.MultipartRequest('POST', Uri.parse(url));
      request.fields['userId'] = userId;
      request.fields['name'] = name;  // Added name field
      request.fields['question'] = question;
      request.fields['description'] = description;

      if (uploadedImg != null && await uploadedImg.exists()) {
        String getContentType(String filePath) {
          final extension = filePath.split('.').last.toLowerCase();
          switch (extension) {
            case 'jpg':
            case 'jpeg':
              return 'image/jpeg';
            case 'png':
              return 'image/png';
            case 'gif':
              return 'image/gif';
            default:
              return 'application/octet-stream';
          }
        }

        final contentType = getContentType(uploadedImg.path);
        
        request.files.add(
          await http.MultipartFile.fromPath(
            'uploaded_img',
            uploadedImg.path,
            contentType: MediaType.parse(contentType),
        ));
      }

      final response = await request.send();
      final responseData = await response.stream.bytesToString();
      final Map<String, dynamic> responseBody = jsonDecode(responseData);

      if (response.statusCode == 201) {
        return {
          'success': true,
          'data': responseBody,
          'message': 'Post created successfully!'
        };
      } else {
        return {
          'success': false,
          'message': responseBody['message'] ?? 'Failed to create post',
          'statusCode': response.statusCode
        };
      }
    } catch (error) {
      return {
        'success': false,
        'message': 'Error creating post: $error'
      };
    }
  }

  Future<void> _submitPost() async {
    // Get current user details
    final String? userId = _authService.getUserUID();
    final Map<String, dynamic>? userDetails = await _authService.getUserDetails();
    
    if (userId == null || userDetails == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('You must be logged in to post')),
      );
      return;
    }

    if (questionController.text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter a question')),
      );
      return;
    }

    if (descriptionController.text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter a description')),
      );
      return;
    }

    setState(() {
      _isLoading = true;
    });

    try {
      final result = await createPost(
        userId: userId,
        name: userDetails['fullName'] ?? 'Anonymous', // Get name from user details
        question: questionController.text,
        description: descriptionController.text,
        uploadedImg: _selectedImage,
      );

      if (result['success'] == true) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Post created successfully!')),
        );
        Navigator.pop(context);
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(result['message'] ?? 'Failed to create post')),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: ${e.toString()}')),
      );
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 1,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.black),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text(
          "Ask Community",
          style: TextStyle(color: Colors.black),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
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
                            icon: const Icon(Icons.close, color: Colors.white),
                            onPressed: _removeImage,
                            style: IconButton.styleFrom(
                              backgroundColor: Colors.black54,
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                  ],
                ),
              ElevatedButton.icon(
                onPressed: _pickImage,
                icon: const Icon(Icons.add_photo_alternate),
                label: Text(_selectedImage == null ? "Add Image" : "Change Image"),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.white,
                  foregroundColor: Colors.black,
                  side: BorderSide(color: Colors.grey.shade300),
                ),
              ),
              const SizedBox(height: 16),
             
              const Text("Your question to the community",
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
              const SizedBox(height: 8),
              TextField(
                controller: questionController,
                maxLength: 200,
                maxLines: 3,
                decoration: const InputDecoration(
                  hintText: "Add a question indicating what's wrong with your crop",
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 24),
              const Text("Description of your problem",
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
              const SizedBox(height: 8),
              TextField(
                controller: descriptionController,
                maxLength: 2500,
                maxLines: 6,
                decoration: const InputDecoration(
                  hintText: "Describe specialities such as change of leaves, root colour, bugs, tears...",
                  border: OutlineInputBorder(),
                ),
              ),
            ],
          ),
        ),
      ),
      bottomNavigationBar: Padding(
        padding: const EdgeInsets.all(16.0),
        child: ElevatedButton(
          onPressed: _isLoading ? null : _submitPost,
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.blue,
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(vertical: 16),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(30),
            ),
          ),
          child: _isLoading 
              ? const CircularProgressIndicator(color: Colors.white)
              : const Text("Send", style: TextStyle(fontSize: 16)),
        ),
      ),
    );
  }
}