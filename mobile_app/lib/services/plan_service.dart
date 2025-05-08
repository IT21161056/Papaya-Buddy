import 'package:cloud_firestore/cloud_firestore.dart';

class PlanService {
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;

  Future<Map<String, dynamic>?> getPlanById(String planId) async {
    try {
      DocumentSnapshot doc =
          await _firestore.collection('subscription_plans').doc(planId).get();
      return doc.data() as Map<String, dynamic>?;
    } catch (e) {
      print("Error fetching plan: $e");
      return null;
    }
  }

  Future<List<Map<String, dynamic>>> getAvailablePlans() async {
    try {
      QuerySnapshot snapshot = await _firestore
          .collection('subscription_plans')
          .where('isActive', isEqualTo: true)
          .get();

      return snapshot.docs.map((doc) {
        final data = doc.data() as Map<String, dynamic>;
        return {
          ...data,
          'id': doc.id, // Include document ID for reference
        };
      }).toList();
    } catch (e) {
      print("Error fetching plans: $e");
      return [];
    }
  }
}
