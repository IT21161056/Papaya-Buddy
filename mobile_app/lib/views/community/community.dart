import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:lucide_icons/lucide_icons.dart';
import 'package:mobile_app/services/auth_services.dart';
import 'package:mobile_app/views/community/ask_community.dart';
import 'package:mobile_app/utils/constants.dart';
import 'dart:convert';

void main() => runApp(CommunityView());

class CommunityView extends StatefulWidget {
  @override
  State<CommunityView> createState() => _CommunityViewState();
}

class _CommunityViewState extends State<CommunityView> {
  String activeTab = 'community';
  String searchQuery = '';
  List<dynamic> posts = [];
  bool isLoading = true;
  String? currentPostId;
  TextEditingController commentController = TextEditingController();
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();
  final AuthService _authService = AuthService();

  @override
  void initState() {
    super.initState();
    fetchCommunityPosts();
  }

  Future<void> fetchCommunityPosts() async {
    try {
      final response = await http.get(
        Uri.parse('${BaseURL.BASE_URL}:5080/api/v1/community'),
      );

      if (response.statusCode == 200) {
        final List<dynamic> fetchedPosts = json.decode(response.body);
        setState(() {
          posts = fetchedPosts;
          isLoading = false;
        });
      } else {
        throw Exception('Failed to load posts');
      }
    } catch (e) {
      setState(() {
        isLoading = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error fetching posts: $e')),
      );
    }
  }

  Future<void> addComment(String postId, String commentText) async {
    final String? userId = _authService.getUserUID();
    final Map<String, dynamic>? userDetails = await _authService.getUserDetails();
    
    try {
      final response = await http.post(
        Uri.parse('${BaseURL.BASE_URL}:5080/api/v1/community/comment/$postId'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'user': userDetails?['fullName'] ?? 'Anonymous',
          'text': commentText,
        }),
      );

      if (response.statusCode == 200) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Comment added successfully!')),
        );
        fetchCommunityPosts(); // Refresh posts to show new comment
        commentController.clear();
      } else {
        throw Exception('Failed to add comment');
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error adding comment: $e')),
      );
    }
  }

  String formatTimeAgo(DateTime postDate) {
    final now = DateTime.now();
    final difference = now.difference(postDate);

    if (difference.inDays > 0) {
      return '${difference.inDays}d';
    } else if (difference.inHours > 0) {
      return '${difference.inHours}h';
    } else if (difference.inMinutes > 0) {
      return '${difference.inMinutes}m';
    } else {
      return 'Just now';
    }
  }

  void navigateToAskCommunity() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => AskCommunityView()),
    ).then((_) => fetchCommunityPosts());
  }

  void _showCommentsDrawer(String postId, List<dynamic> comments) {
  setState(() {
    currentPostId = postId;
  });

  showModalBottomSheet(
    context: context,
    isScrollControlled: true,
    builder: (context) {
      return DraggableScrollableSheet(
        initialChildSize: 0.6,
        minChildSize: 0.4,
        maxChildSize: 0.8,
        expand: false,
        builder: (context, scrollController) {
          return Container(
            padding: EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
            child: Column(
              children: [
                Container(
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(
                    color: Colors.grey[300],
                    borderRadius: BorderRadius.circular(2)),
                ),
                SizedBox(height: 16),
                Text(
                  'Comments',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 16),
                Expanded(
                  child: ListView.builder(
                    controller: scrollController,
                    itemCount: comments.length,
                    itemBuilder: (context, index) {
                      final comment = comments[index];
                      final commentDate = DateTime.parse(comment['createdAt']);
                      final timeAgo = formatTimeAgo(commentDate);
                      
                      return Padding(
                        padding: EdgeInsets.symmetric(vertical: 8),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                CircleAvatar(
                                  child: Icon(Icons.person),
                                ),
                                SizedBox(width: 8),
                                Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      comment['user'] ?? 'Anonymous',
                                      style: TextStyle(
                                        fontWeight: FontWeight.bold),
                                    ),
                                    Text(
                                      timeAgo,
                                      style: TextStyle(
                                        fontSize: 12,
                                        color: Colors.grey),
                                    ),
                                  ],
                                ),
                              ],
                            ),
                            SizedBox(height: 4),
                            Padding(
                              padding: EdgeInsets.only(left: 40),
                              child: Text(comment['text']),
                            ),
                            Divider(),
                          ],
                        ),
                      );
                    },
                  ),
                ),
                Padding(
                  padding: EdgeInsets.only(
                    bottom: MediaQuery.of(context).viewInsets.bottom),
                  child: Row(
                    children: [
                      Expanded(
                        child: TextField(
                          controller: commentController,
                          decoration: InputDecoration(
                            hintText: 'Write a comment...',
                            border: OutlineInputBorder(),
                            contentPadding: EdgeInsets.symmetric(
                                horizontal: 12, vertical: 12),
                          ),
                        ),
                      ),
                      SizedBox(width: 8),
                      IconButton(
                        icon: Icon(Icons.send, color: Colors.blue),
                        onPressed: () {
                          if (commentController.text.isNotEmpty && 
                              currentPostId != null) {
                            addComment(currentPostId!, commentController.text);
                          }
                        },
                      ),
                    ],
                  ),
                ),
              ],
            ),
          );
        },
      );
    },
  );
}

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        key: _scaffoldKey,
        backgroundColor: Colors.grey[100],
        appBar: AppBar(
          backgroundColor: Colors.white,
          elevation: 1,
          title: TextField(
            onChanged: (value) {
              setState(() {
                searchQuery = value.toLowerCase();
              });
            },
            decoration: InputDecoration(
              hintText: 'Search in Community',
              prefixIcon: Icon(Icons.search, color: Colors.grey),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(30),
                borderSide: BorderSide.none,
              ),
              filled: true,
              fillColor: Colors.grey[200],
            ),
          ),
          actions: [
            IconButton(
              icon: Icon(Icons.notifications, color: Colors.grey),
              onPressed: () {},
            ),
            IconButton(
              icon: Icon(Icons.more_horiz, color: Colors.grey),
              onPressed: () {},
            ),
          ],
        ),
        body: isLoading
            ? Center(child: CircularProgressIndicator())
            : Padding(
                padding: EdgeInsets.all(10.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Filter by', style: TextStyle(fontWeight: FontWeight.bold)),
                    SizedBox(height: 10),
                    Row(
                      children: [
                        Chip(label: Text('Papaya')),
                        SizedBox(width: 5),
                        Chip(label: Text('Latest')),
                      ],
                    ),
                    Expanded(
                      child: ListView.builder(
                        itemCount: posts.length,
                        itemBuilder: (context, index) {
                          final post = posts[index];
                          final postDate = DateTime.parse(post['createdAt']);
                          final timeAgo = formatTimeAgo(postDate);
                          final comments = post['comments'] ?? [];

                          if (searchQuery.isNotEmpty &&
                              !post['question'].toLowerCase().contains(searchQuery)) {
                            return SizedBox.shrink();
                          }

                          return Card(
                            margin: EdgeInsets.symmetric(vertical: 8),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                if (post['uploaded_img'] != null)
                                  ClipRRect(
                                    borderRadius: BorderRadius.vertical(
                                      top: Radius.circular(10),
                                    ),
                                    child: Image.network(
                                      post['uploaded_img'],
                                      height: 180,
                                      width: double.infinity,
                                      fit: BoxFit.cover,
                                      errorBuilder: (context, error, stackTrace) {
                                        return Container(
                                          height: 180,
                                          color: Colors.grey[200],
                                          child: Icon(Icons.broken_image),
                                        );
                                      },
                                    ),
                                  ),
                                Padding(
                                  padding: EdgeInsets.all(10),
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      Row(
                                        children: [
                                          CircleAvatar(
                                            child: Icon(Icons.person),
                                          ),
                                          SizedBox(width: 10),
                                          Column(
                                            crossAxisAlignment:
                                                CrossAxisAlignment.start,
                                            children: [
                                              Text(
                                                post['name'] ?? 'Anonymous',
                                                style: TextStyle(
                                                  fontWeight: FontWeight.bold),
                                              ),
                                              Text(
                                                'Sri Lanka • $timeAgo',
                                                style: TextStyle(
                                                  color: Colors.grey),
                                              ),
                                            ],
                                          ),
                                        ],
                                      ),
                                      SizedBox(height: 10),
                                      Text(
                                        post['question'],
                                        style: TextStyle(
                                          fontWeight: FontWeight.bold),
                                      ),
                                      SizedBox(height: 5),
                                      Text(
                                        post['description'],
                                        style: TextStyle(color: Colors.grey),
                                      ),
                                      SizedBox(height: 10),
                                      Row(
                                        children: [
                                          IconButton(
                                            icon: Icon(LucideIcons.messageSquare),
                                            onPressed: () => _showCommentsDrawer(
                                                post['_id'], comments),
                                          ),
                                          Text(comments.length.toString()),
                                          SizedBox(width: 20),
                                          IconButton(
                                            icon: Icon(LucideIcons.thumbsUp),
                                            onPressed: () {},
                                          ),
                                          Text('0'),
                                        ],
                                      ),
                                    ],
                                  ),
                                ),
                              ],
                            ),
                          );
                        },
                      ),
                    ),
                  ],
                ),
              ),
        floatingActionButton: FloatingActionButton.extended(
          onPressed: navigateToAskCommunity,
          label: Text('Ask Community'),
          icon: Icon(Icons.send),
          backgroundColor: Colors.blue,
        ),
      ),
    );
  }
}