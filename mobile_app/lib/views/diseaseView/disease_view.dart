import 'package:flutter/material.dart';
import 'package:carousel_slider/carousel_slider.dart';

class DiseaseView extends StatelessWidget {
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
    'assets/r2.jpg',
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
            child:
                Image.asset('assets/icons/disease.png', width: 28, height: 28),
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            CarouselSlider(
              options: CarouselOptions(
                height: 240,
                enlargeCenterPage: true,
                enableInfiniteScroll: true,
                autoPlay: true,
                scrollDirection: Axis.horizontal,
              ),
              items: imagePaths.map((path) {
                return Container(
                  margin: EdgeInsets.symmetric(horizontal: 5.0),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(15.0),
                    image: DecorationImage(
                      image: AssetImage(path),
                      fit: BoxFit.cover,
                    ),
                  ),
                );
              }).toList(),
            ),
            SizedBox(
              height: 16,
            ),
            Container(
              padding: EdgeInsets.all(8.0),
              decoration: BoxDecoration(
                color: Colors.green[100],
                borderRadius: BorderRadius.circular(8.0),
              ),
              child: Row(
                children: [
                  Icon(Icons.info, color: Colors.green),
                  SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      causedBy,
                      style: TextStyle(
                        fontSize: 15,
                        fontFamily: 'Moon',
                      ),
                    ),
                  ),
                ],
              ),
            ),
            SizedBox(height: 8),
            DiseaseButton(title: 'What caused it?'),
            DiseaseButton(title: 'Symptoms'),
            DiseaseButton(title: 'Treatment Instructions'),
          ],
        ),
      ),
    );
  }
}

class DiseaseButton extends StatelessWidget {
  final String title;
  DiseaseButton({required this.title});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 12),
      child: ElevatedButton(
        onPressed: () {},
        style: ElevatedButton.styleFrom(
          minimumSize: Size(double.infinity, 60),
          backgroundColor: Colors.white,
          foregroundColor: const Color.fromARGB(255, 25, 99, 28),
          textStyle: TextStyle(fontSize: 18),
          side: BorderSide(color: Color.fromARGB(255, 30, 246, 199)),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15),
          ),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(title),
            Icon(Icons.arrow_forward, color: Color(0xFF15B392)),
          ],
        ),
      ),
    );
  }
}
