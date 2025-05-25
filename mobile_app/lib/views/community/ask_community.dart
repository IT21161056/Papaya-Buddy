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

class _AskCommunityScreenState extends State<AskCommunityView>
    with SingleTickerProviderStateMixin {
  final TextEditingController questionController = TextEditingController();
  final TextEditingController descriptionController = TextEditingController();
  final AuthService _authService = AuthService();
  final FocusNode _questionFocusNode = FocusNode();
  final FocusNode _descriptionFocusNode = FocusNode();

  File? _selectedImage;
  bool _isLoading = false;
  bool _isImageLoading = false;
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: Duration(milliseconds: 300),
      vsync: this,
    );
    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeInOut),
    );
    _animationController.forward();
  }

  @override
  void dispose() {
    _animationController.dispose();
    _questionFocusNode.dispose();
    _descriptionFocusNode.dispose();
    questionController.dispose();
    descriptionController.dispose();
    super.dispose();
  }

  Future<void> _pickImage() async {
    setState(() {
      _isImageLoading = true;
    });

    try {
      final picker = ImagePicker();
      final pickedFile = await picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 85,
      );

      if (pickedFile != null) {
        setState(() {
          _selectedImage = File(pickedFile.path);
        });
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error selecting image: $e'),
          backgroundColor: Colors.red[400],
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        ),
      );
    } finally {
      setState(() {
        _isImageLoading = false;
      });
    }
  }

  void _showImageOptions() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (context) {
        return Container(
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.1),
                blurRadius: 10,
                offset: Offset(0, -2),
              ),
            ],
          ),
          child: SafeArea(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  padding: EdgeInsets.symmetric(vertical: 12),
                  child: Container(
                    width: 40,
                    height: 4,
                    decoration: BoxDecoration(
                      color: Colors.grey[300],
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),
                ),
                Padding(
                  padding: EdgeInsets.all(20),
                  child: Column(
                    children: [
                      Text(
                        'Add Photo',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.w600,
                          color: Color.fromRGBO(26, 26, 26, 1),
                        ),
                      ),
                      SizedBox(height: 20),
                      _buildImageOption(
                        icon: Icons.photo_library_outlined,
                        title: 'Choose from Gallery',
                        subtitle: 'Select an existing photo',
                        onTap: () {
                          Navigator.pop(context);
                          _pickImage();
                        },
                      ),
                      SizedBox(height: 12),
                      _buildImageOption(
                        icon: Icons.camera_alt_outlined,
                        title: 'Take Photo',
                        subtitle: 'Use camera to capture',
                        onTap: () {
                          Navigator.pop(context);
                          _takePicture();
                        },
                      ),
                      if (_selectedImage != null) ...[
                        SizedBox(height: 12),
                        _buildImageOption(
                          icon: Icons.delete_outline,
                          title: 'Remove Photo',
                          subtitle: 'Delete current image',
                          onTap: () {
                            Navigator.pop(context);
                            _removeImage();
                          },
                          isDestructive: true,
                        ),
                      ],
                      SizedBox(height: 20),
                    ],
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildImageOption({
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
    bool isDestructive = false,
  }) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(12),
      child: Container(
        padding: EdgeInsets.all(16),
        decoration: BoxDecoration(
          border: Border.all(color: Colors.grey[200]!),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          children: [
            CircleAvatar(
              backgroundColor: isDestructive
                  ? Colors.red.withOpacity(0.1)
                  : Colors.blue.withOpacity(0.1),
              child: Icon(
                icon,
                color: isDestructive ? Colors.red : Colors.blue,
                size: 20,
              ),
            ),
            SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: TextStyle(
                      fontWeight: FontWeight.w600,
                      color: isDestructive
                          ? Colors.red
                          : Color.fromRGBO(26, 26, 26, 1),
                    ),
                  ),
                  SizedBox(height: 2),
                  Text(
                    subtitle,
                    style: TextStyle(
                      fontSize: 12,
                      color: Color.fromRGBO(100, 116, 139, 1),
                    ),
                  ),
                ],
              ),
            ),
            Icon(
              Icons.arrow_forward_ios,
              size: 16,
              color: Colors.grey[400],
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _takePicture() async {
    setState(() {
      _isImageLoading = true;
    });

    try {
      final picker = ImagePicker();
      final pickedFile = await picker.pickImage(
        source: ImageSource.camera,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 85,
      );

      if (pickedFile != null) {
        setState(() {
          _selectedImage = File(pickedFile.path);
        });
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error taking picture: $e'),
          backgroundColor: Colors.red[400],
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        ),
      );
    } finally {
      setState(() {
        _isImageLoading = false;
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
    final String url = '${BaseURL.BASE_URL}/api/v1/community';

    try {
      final request = http.MultipartRequest('POST', Uri.parse(url));
      request.fields['userId'] = userId;
      request.fields['name'] = name;
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

        request.files.add(await http.MultipartFile.fromPath(
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
      return {'success': false, 'message': 'Error creating post: $error'};
    }
  }

  Future<void> _submitPost() async {
    // Validation
    if (questionController.text.trim().isEmpty) {
      _showValidationError('Please enter a question');
      _questionFocusNode.requestFocus();
      return;
    }

    if (descriptionController.text.trim().isEmpty) {
      _showValidationError('Please enter a description');
      _descriptionFocusNode.requestFocus();
      return;
    }

    // Get current user details
    final String? userId = _authService.getUserUID();
    final Map<String, dynamic>? userDetails =
        await _authService.getUserDetails();

    if (userId == null || userDetails == null) {
      _showValidationError('You must be logged in to post');
      return;
    }

    setState(() {
      _isLoading = true;
    });

    try {
      final result = await createPost(
        userId: userId,
        name: userDetails['fullName'] ?? 'Anonymous',
        question: questionController.text.trim(),
        description: descriptionController.text.trim(),
        uploadedImg: _selectedImage,
      );

      if (result['success'] == true) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Row(
              children: [
                Icon(Icons.check_circle, color: Colors.white),
                SizedBox(width: 12),
                Text('Post created successfully!'),
              ],
            ),
            backgroundColor: Colors.green[400],
            behavior: SnackBarBehavior.floating,
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
          ),
        );
        Navigator.pop(context);
      } else {
        _showValidationError(result['message'] ?? 'Failed to create post');
      }
    } catch (e) {
      _showValidationError('Error: ${e.toString()}');
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  void _showValidationError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Row(
          children: [
            Icon(Icons.error_outline, color: Colors.white),
            SizedBox(width: 12),
            Expanded(child: Text(message)),
          ],
        ),
        backgroundColor: Colors.red[400],
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      ),
    );
  }

  Widget _buildImagePreview() {
    if (_selectedImage == null && !_isImageLoading) return SizedBox.shrink();

    return FadeTransition(
      opacity: _fadeAnimation,
      child: Container(
        margin: EdgeInsets.only(bottom: 20),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.1),
              blurRadius: 10,
              offset: Offset(0, 2),
            ),
          ],
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(16),
          child: Container(
            height: 220,
            width: double.infinity,
            child: _isImageLoading
                ? Container(
                    color: Colors.grey[100],
                    child: Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          CircularProgressIndicator(
                            strokeWidth: 2,
                            valueColor:
                                AlwaysStoppedAnimation<Color>(Colors.blue),
                          ),
                          SizedBox(height: 12),
                          Text(
                            'Loading image...',
                            style: TextStyle(
                              color: Color.fromRGBO(100, 116, 139, 1),
                            ),
                          ),
                        ],
                      ),
                    ),
                  )
                : Stack(
                    fit: StackFit.expand,
                    children: [
                      Image.file(
                        _selectedImage!,
                        fit: BoxFit.cover,
                      ),
                      Positioned(
                        top: 12,
                        right: 12,
                        child: Container(
                          decoration: BoxDecoration(
                            color: Colors.black.withOpacity(0.6),
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: IconButton(
                            icon: Icon(Icons.close,
                                color: Colors.white, size: 20),
                            padding: EdgeInsets.all(8),
                            constraints: BoxConstraints(),
                            onPressed: _removeImage,
                          ),
                        ),
                      ),
                    ],
                  ),
          ),
        ),
      ),
    );
  }

  Widget _buildTextField({
    required TextEditingController controller,
    required FocusNode focusNode,
    required String label,
    required String hint,
    required int maxLength,
    required int maxLines,
    bool isRequired = true,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text(
              label,
              style: TextStyle(
                fontWeight: FontWeight.w600,
                fontSize: 16,
                color: Color.fromRGBO(26, 26, 26, 1),
              ),
            ),
            if (isRequired) ...[
              SizedBox(width: 4),
              Text(
                '*',
                style: TextStyle(
                  color: Colors.red,
                  fontSize: 16,
                ),
              ),
            ],
          ],
        ),
        SizedBox(height: 8),
        Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(12),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.02),
                blurRadius: 4,
                offset: Offset(0, 2),
              ),
            ],
          ),
          child: TextField(
            controller: controller,
            focusNode: focusNode,
            maxLength: maxLength,
            maxLines: maxLines,
            style: TextStyle(
              color: Color.fromRGBO(26, 26, 26, 1),
            ),
            decoration: InputDecoration(
              hintText: hint,
              hintStyle: TextStyle(
                color: Color.fromRGBO(100, 116, 139, 1),
              ),
              filled: true,
              fillColor: Colors.white,
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: BorderSide(color: Colors.grey[200]!),
              ),
              enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: BorderSide(color: Colors.grey[200]!),
              ),
              focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: BorderSide(color: Colors.blue, width: 2),
              ),
              contentPadding: EdgeInsets.all(16),
              counterStyle: TextStyle(
                color: Color.fromRGBO(100, 116, 139, 1),
                fontSize: 12,
              ),
            ),
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color.fromRGBO(248, 250, 252, 1),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        shadowColor: Colors.black.withOpacity(0.1),
        leading: IconButton(
          icon: Icon(
            Icons.arrow_back,
            color: Color.fromRGBO(26, 26, 26, 1),
          ),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text(
          "Ask Community",
          style: TextStyle(
            color: Color.fromRGBO(26, 26, 26, 1),
            fontWeight: FontWeight.w600,
          ),
        ),
        centerTitle: true,
      ),
      body: FadeTransition(
        opacity: _fadeAnimation,
        child: SingleChildScrollView(
          padding: EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Image section
              _buildImagePreview(),

              // Add image button
              Container(
                width: double.infinity,
                height: 56,
                margin: EdgeInsets.only(bottom: 32),
                child: OutlinedButton.icon(
                  onPressed: _showImageOptions,
                  icon: Icon(
                    _selectedImage == null
                        ? Icons.add_photo_alternate_outlined
                        : Icons.edit_outlined,
                    size: 20,
                  ),
                  label: Text(
                    _selectedImage == null ? "Add Photo" : "Change Photo",
                    style: TextStyle(
                      fontWeight: FontWeight.w500,
                      fontSize: 16,
                    ),
                  ),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: Colors.blue,
                    side: BorderSide(color: Colors.blue.withOpacity(0.3)),
                    backgroundColor: Colors.blue.withOpacity(0.05),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                ),
              ),

              // Question field
              _buildTextField(
                controller: questionController,
                focusNode: _questionFocusNode,
                label: "Your question to the community",
                hint: "What's wrong with your crop? Be specific...",
                maxLength: 200,
                maxLines: 3,
              ),

              SizedBox(height: 24),

              // Description field
              _buildTextField(
                controller: descriptionController,
                focusNode: _descriptionFocusNode,
                label: "Description of your problem",
                hint:
                    "Describe details like leaf changes, root color, bugs, damage...",
                maxLength: 2500,
                maxLines: 6,
              ),

              SizedBox(height: 32),

              // Help text
              Container(
                padding: EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.blue.withOpacity(0.05),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.blue.withOpacity(0.1)),
                ),
                child: Row(
                  children: [
                    Icon(
                      Icons.lightbulb_outline,
                      color: Colors.blue,
                      size: 20,
                    ),
                    SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'Tip: Include clear photos and detailed descriptions to get better help from the community.',
                        style: TextStyle(
                          color: Colors.blue[700],
                          fontSize: 14,
                        ),
                      ),
                    ),
                  ],
                ),
              ),

              SizedBox(height: 100), // Space for bottom button
            ],
          ),
        ),
      ),
      bottomNavigationBar: Container(
        padding: EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.white,
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.05),
              blurRadius: 10,
              offset: Offset(0, -2),
            ),
          ],
        ),
        child: SafeArea(
          child: Container(
            height: 56,
            width: double.infinity,
            child: ElevatedButton(
              onPressed: _isLoading ? null : _submitPost,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blue,
                foregroundColor: Colors.white,
                disabledBackgroundColor: Colors.grey[300],
                elevation: 0,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
              ),
              child: _isLoading
                  ? Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            color: Colors.white,
                            strokeWidth: 2,
                          ),
                        ),
                        SizedBox(width: 12),
                        Text(
                          "Posting...",
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    )
                  : Text(
                      "Post Question",
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
            ),
          ),
        ),
      ),
    );
  }
}
