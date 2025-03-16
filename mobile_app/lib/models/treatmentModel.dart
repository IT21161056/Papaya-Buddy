
class Treatment {
  final String id;
  final String method;
  final String description;
  final String diseaseId;
  final List<String> treatmentType;
  final List<String> effectiveness;
  final String sideEffects;
  final String precautions;
  final DateTime createdAt;

  Treatment({
    required this.id,
    required this.method,
    required this.description,
    required this.diseaseId,
    required this.treatmentType,
    required this.effectiveness,
    required this.sideEffects,
    required this.precautions,
    required this.createdAt,
  });

  // Convert Treatment instance to a map
  Map<String, dynamic> toMap() {
    return {
      '_id': id,
      'method': method,
      'description': description,
      'diseaseId': diseaseId,
      'treatment_type': treatmentType,
      'effectiveness': effectiveness,
      'side_effects': sideEffects,
      'precautions': precautions,
      'created_at': createdAt,
    };
  }

  // Create Treatment instance from a map
  factory Treatment.fromJson(Map<String, dynamic> map) {
    return Treatment(
      id: map['_id'].toString(),
      method: map['method'] as String,
      description: map['description'] as String,
      diseaseId: map['diseaseId'] as String,
      treatmentType: List<String>.from(map['treatment_type']),
      effectiveness: List<String>.from(map['effectiveness']),
      sideEffects: map['side_effects'] as String,
      precautions: map['precautions'] as String,
      createdAt: map['created_at'] as DateTime,
    );
  }

  // Create a copy of this Treatment with the given field values updated
  Treatment copyWith({
    String? id,
    String? method,
    String? description,
    String? diseaseId,
    List<String>? treatmentType,
    List<String>? effectiveness,
    String? sideEffects,
    String? precautions,
    DateTime? createdAt,
  }) {
    return Treatment(
      id: id ?? this.id,
      method: method ?? this.method,
      description: description ?? this.description,
      diseaseId: diseaseId ?? this.diseaseId,
      treatmentType: treatmentType ?? this.treatmentType,
      effectiveness: effectiveness ?? this.effectiveness,
      sideEffects: sideEffects ?? this.sideEffects,
      precautions: precautions ?? this.precautions,
      createdAt: createdAt ?? this.createdAt,
    );
  }
}
