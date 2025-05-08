import 'package:flutter/material.dart';
import 'package:lucide_icons/lucide_icons.dart';
import 'package:mobile_app/views/community/ask_community.dart';

void main() => runApp(CommunityView());

class CommunityView extends StatefulWidget {
  @override
  State<CommunityView> createState() => _CommunityViewState();
}

class _CommunityViewState extends State<CommunityView> {
  String activeTab = 'community';
  String searchQuery = '';

  final List<Map<String, dynamic>> posts = [
    {
      'id': 1,
      'author': 'Pasindu',
      'location': 'Sri Lanka',
      'timeAgo': '30 m',
      'category': 'Papaya',
      'title': 'Help identifying problem with my Papaya',
      'description':
          'Flowers are not becoming fruit, it is drying before fruiting and dropping.',
      'image': 'assets/r1.jpg',
      'answers': 0,
      'likes': 0,
      'dislikes': 0,
    }
  ];

  void navigateToAskCommunity() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => AskCommunityView()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        backgroundColor: Colors.grey[100],
        appBar: AppBar(
          backgroundColor: Colors.white,
          elevation: 1,
          title: TextField(
            onChanged: (value) {
              setState(() {
                searchQuery = value;
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
                onPressed: () {}),
            IconButton(
                icon: Icon(Icons.more_horiz, color: Colors.grey),
                onPressed: () {}),
          ],
        ),
        body: Padding(
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
                  Chip(label: Text('Popular')),
                ],
              ),
              Expanded(
                child: ListView.builder(
                  itemCount: posts.length,
                  itemBuilder: (context, index) {
                    final post = posts[index];
                    return Card(
                      margin: EdgeInsets.symmetric(vertical: 10),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10)),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          ClipRRect(
                            borderRadius:
                                BorderRadius.vertical(top: Radius.circular(10)),
                            child: Image.asset(
                                post['image'], // Changed to Image.asset
                                height: 180,
                                width: double.infinity,
                                fit: BoxFit.cover),
                          ),
                          Padding(
                            padding: EdgeInsets.all(10),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  children: [
                                    CircleAvatar(child: Icon(Icons.person)),
                                    SizedBox(width: 10),
                                    Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        Text(post['author'],
                                            style: TextStyle(
                                                fontWeight: FontWeight.bold)),
                                        Text(
                                            '${post['location']} • ${post['timeAgo']} • ${post['category']}',
                                            style:
                                                TextStyle(color: Colors.grey)),
                                      ],
                                    ),
                                  ],
                                ),
                                SizedBox(height: 10),
                                Text(post['title'],
                                    style:
                                        TextStyle(fontWeight: FontWeight.bold)),
                                SizedBox(height: 5),
                                Text(post['description'],
                                    style: TextStyle(color: Colors.grey)),
                              ],
                            ),
                          )
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
