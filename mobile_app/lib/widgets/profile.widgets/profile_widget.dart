import 'dart:convert';
import 'dart:io';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:http/http.dart' as http;
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter/widgets.dart';
import 'package:mobile_app/theme/colors.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:image_picker/image_picker.dart';

class ProfileCard extends StatefulWidget {
  final String userName;
  final String description;
  final String profilePicture;
  final Function(String newImageUrl)? onProfilePictureUpdated;

  const ProfileCard({
    super.key,
    required this.userName,
    required this.description,
    required this.profilePicture,
    this.onProfilePictureUpdated,
  });

  State<ProfileCard> createState() => _ProfileCardState();
}

class _ProfileCardState extends State<ProfileCard> {
  final ImagePicker _picker = ImagePicker();
  bool _isUploading = false;
  String? _tempProfilePicture;

  Future<void> _uploadProfilePicture() async {
    try {
      // Pick image from gallery
      final XFile? pickedFile = await _picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 800,
        maxHeight: 800,
        imageQuality: 80,
      );

      if (pickedFile == null) return;

      setState(() {
        _isUploading = true;
      });

      // Get current user
      User? user = FirebaseAuth.instance.currentUser;
      if (user == null) {
        throw Exception("User not authenticated");
      }

      // Prepare file for upload
      File imageFile = File(pickedFile.path);
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('http://localhost:5080/api/upload'),
      );

      request.files
          .add(await http.MultipartFile.fromPath('file', imageFile.path));

      var response = await request.send();
      if (response.statusCode != 200) {
        throw Exception("Failed to upload image");
      }

      var responseData = await response.stream.bytesToString();
      var jsonResponse = json.decode(responseData);

      if (!(jsonResponse['success'] ?? false)) {
        throw Exception(jsonResponse['message'] ?? "Upload failed");
      }

      String cloudinaryUrl = jsonResponse['data']['url'];

      // Update Firestore user record with new profile picture
      await FirebaseFirestore.instance
          .collection('users')
          .doc(user.uid)
          .update({
        'image_url': cloudinaryUrl,
      });

      // Update UI
      if (widget.onProfilePictureUpdated != null) {
        widget.onProfilePictureUpdated!(cloudinaryUrl);
      }

      setState(() {
        _tempProfilePicture = cloudinaryUrl;
        _isUploading = false;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Profile picture updated successfully!')),
      );
    } catch (e) {
      setState(() {
        _isUploading = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to update profile picture: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final currentProfilePicture = _tempProfilePicture ?? widget.profilePicture;
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 5,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        children: [
          Stack(
            children: [
              CircleAvatar(
                radius: 40,
                backgroundImage:
                    _isUploading ? null : NetworkImage(currentProfilePicture),
                child: _isUploading ? const CircularProgressIndicator() : null,
              ),

              // Camera icon Button
              Positioned(
                  right: 0,
                  bottom: 0,
                  child: GestureDetector(
                    onTap: _isUploading ? null : _uploadProfilePicture,
                    child: CircleAvatar(
                      radius: 14,
                      backgroundColor: AppColors.primary,
                      child: const Icon(Icons.camera_alt,
                          size: 16, color: Colors.white),
                    ),
                  )),
            ],
          ),
          const SizedBox(width: 20),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  widget.userName,
                  style: TextStyle(
                      fontWeight: FontWeight.w700,
                      fontSize: 20,
                      color: Color.fromRGBO(26, 26, 26, 1)),
                ),
                const SizedBox(height: 4),
                Text(
                  widget.description,
                  style: TextStyle(
                      fontSize: 14, color: Color.fromRGBO(100, 116, 139, 1)),
                ),
                const SizedBox(height: 12),
                SizedBox(
                  child: TextButton(
                      onPressed: () {},
                      style: TextButton.styleFrom(
                        backgroundColor: AppColors.primary.withOpacity(0.1),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(50),
                        ),
                      ),
                      child: Padding(
                        padding: EdgeInsets.symmetric(vertical: 6),
                        child: IntrinsicWidth(
                          child: Row(
                            children: [
                              SvgPicture.asset(
                                'assets/icons/hugeicons_profile.svg',
                                height: 18,
                                width: 18,
                                color: AppColors.primary,
                              ),
                              const SizedBox(width: 8),
                              Text(
                                "Edit Profile",
                                style: TextStyle(
                                  fontSize: 12,
                                  fontWeight: FontWeight.w500,
                                  color: AppColors.primary,
                                ),
                              ),
                            ],
                          ),
                        ),
                      )),
                )
              ],
            ),
          ),
        ],
      ),
    );
  }
}
