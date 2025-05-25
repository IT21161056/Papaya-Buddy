import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:mobile_app/utils/constants.dart';

class CommunityService {
  static Future<Map<String, dynamic>> createPost({
    required String userId,
    required String question,
    required String description,
    String name = '',
    File? uploadedImg,
  }) async {
    final String url = '${BaseURL.BASE_URL}/api/v1/community';

    try {
      final request = http.MultipartRequest('POST', Uri.parse(url));

      request.fields['userId'] = userId;
      request.fields['question'] = question;
      request.fields['description'] = description;
      request.fields['name'] = name;

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
}
