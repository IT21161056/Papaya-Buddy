// Model classes
class Disease {
  final String id;
  final String name;
  final String affectedArea;
  final String diseaseType;
  final String description;

  Disease({
    required this.id,
    required this.name,
    required this.affectedArea,
    required this.diseaseType,
    required this.description,
  });

  factory Disease.fromJson(Map<String, dynamic> json) {
    return Disease(
      id: json['_id'],
      name: json['name'],
      affectedArea: json['affected_area'],
      diseaseType: json['disease_type'],
      description: json['description'],
    );
  }
}

class Treatment {
  final String id;
  final String method;
  final String description;
  final Disease disease;
  final String treatmentType;
  final String effectiveness;
  final String sideEffects;
  final String precautions;
  final DateTime createdAt;

  Treatment({
    required this.id,
    required this.method,
    required this.description,
    required this.disease,
    required this.treatmentType,
    required this.effectiveness,
    required this.sideEffects,
    required this.precautions,
    required this.createdAt,
  });

  factory Treatment.fromJson(Map<String, dynamic> json) {
    return Treatment(
      id: json['_id'],
      method: json['method'],
      description: json['description'],
      disease: Disease.fromJson(json['diseaseId']),
      treatmentType: json['treatment_type'],
      effectiveness: json['effectiveness'],
      sideEffects: json['side_effects'],
      precautions: json['precautions'],
      createdAt: DateTime.parse(json['created_at']),
    );
  }
}
