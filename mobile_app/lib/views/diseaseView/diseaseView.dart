import 'package:flutter/material.dart';
import 'package:carousel_slider/carousel_slider.dart';

class Diseaseviewwidget extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(primarySwatch: Colors.green),
      home: DiseaseScreen(),
    );
  }
}

class DiseaseScreen extends StatelessWidget {
  final List<String> imagePaths = [
    'assets/r2.jpg',
    'assets/icons8-virus-48.png',
  ];

  final String causedBy =
      "Caused by a single-stranded RNA virus belonging to the Potyvirus genus in the family Potyviridae.";

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        leading: IconButton(icon: Icon(Icons.arrow_back), onPressed: () {}),
        title: Text('Disease'),
        actions: [
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Image.asset('assets/icons8-virus-48.png',
                width: 28, height: 28),
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            CarouselSlider(
              options: CarouselOptions(
                height: 250,
                enlargeCenterPage: true,
                enableInfiniteScroll: true,
                autoPlay: false,
                scrollDirection: Axis.horizontal,
              ),
              items: imagePaths.map((path) {
                return Container(
                  margin: EdgeInsets.symmetric(horizontal: 5.0),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(15.0),
                    image: DecorationImage(
                        image: AssetImage(path), fit: BoxFit.cover),
                  ),
                );
              }).toList(),
            ),
            SizedBox(height: 16),
            Container(
              padding: EdgeInsets.all(8.0),
              decoration: BoxDecoration(
                color: Color(0xFFE8F5E9), // Light green background color
                borderRadius: BorderRadius.circular(8.0),
              ),
              child: Row(
                children: [
                  Icon(Icons.info, color: Color(0xFF18836D)),
                  SizedBox(width: 8),
                  Expanded(
                    child: Opacity(
                      opacity: 0.7,
                      child: Text(
                        'Caused by a single-stranded RNA virus belonging to the Potyvirus genus in the family Potyviridae.',
                        style: TextStyle(
                          fontSize: 15,
                          fontFamily: 'Moon',
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            SizedBox(height: 12),
            DiseaseButton(
              title: 'What caused it?',
              imagePath: 'assets/r2.jpg',
            ),
            DiseaseButton(
              title: 'Treatment Instructions',
              imagePath: 'assets/r2.jpg',
            ),
          ],
        ),
      ),
    );
  }
}

class DiseaseButton extends StatelessWidget {
  final String title;
  final String imagePath;

  DiseaseButton({required this.title, required this.imagePath});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 12),
      child: ElevatedButton(
        onPressed: () {},
        style: ElevatedButton.styleFrom(
          minimumSize: Size(double.infinity, 90),
          backgroundColor: Colors.white,
          foregroundColor: const Color.fromARGB(255, 25, 99, 28),
          textStyle: TextStyle(fontSize: 18, fontFamily: 'Moon'),
          side: BorderSide(color: Color(0xFFFFFF)),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15),
          ),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Row(
              children: [
                Image.asset(imagePath, width: 48, height: 48),
                SizedBox(width: 10),
                Text(title, style: TextStyle(fontFamily: 'Moon')),
              ],
            ),
            Icon(Icons.arrow_forward, color: Color(0xFF15B392)),
          ],
        ),
      ),
    );
  }
}
