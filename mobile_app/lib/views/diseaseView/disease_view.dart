import 'package:flutter/material.dart';

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
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        leading: IconButton(icon: Icon(Icons.arrow_back), onPressed: () {}),
        title: Text('Disease'),
        actions: [
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Image.asset('assets/icons/icons8-virus-48.png',
                width: 28, height: 28),
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Expanded(
                  child: Column(
                    children: [
                      Image.asset('assets/icons/icons8-virus-48.png',
                          width: 140, height: 140),
                      SizedBox(height: 8),
                      Container(
                        padding:
                            EdgeInsets.symmetric(vertical: 8, horizontal: 8),
                        decoration: BoxDecoration(
                          color: Color(0xFF73EC8B),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          'Your Image',
                          style: TextStyle(color: Colors.black45, fontSize: 16),
                        ),
                      ),
                    ],
                  ),
                ),
                SizedBox(width: 16),
                Expanded(
                  child: Column(
                    children: [
                      Image.asset('assets/icons/icons8-virus-48.png',
                          width: 140, height: 140),
                      SizedBox(height: 8),
                      Container(
                        padding:
                            EdgeInsets.symmetric(vertical: 8, horizontal: 8),
                        decoration: BoxDecoration(
                          color: Color(0xFF73EC8B),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          'Your Image',
                          style: TextStyle(color: Colors.black45, fontSize: 16),
                        ),
                      ),
                    ],
                  ),
                )
              ],
            ),
            SizedBox(
              height: 24,
            )
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
