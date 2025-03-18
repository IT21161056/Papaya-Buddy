class Disease {
  final String id;
  final String name;
  final String description;

  Disease({
    required this.id,
    required this.name,
    required this.description,
  });

  factory Disease.fromJson(Map<String, dynamic> json) {
    return Disease(
      id: json['_id'],
      name: json['name'],
      description: json['description'],
    );
  }
}

class Prediction {
  final String id;
  final String uploadedImgUrl;
  final String userId;
  final Disease disease;
  final DateTime createdAt;

  Prediction({
    required this.id,
    required this.uploadedImgUrl,
    required this.userId,
    required this.disease,
    required this.createdAt,
  });

  factory Prediction.fromJson(Map<String, dynamic> json) {
    return Prediction(
      id: json['_id'],
      uploadedImgUrl: json['uploaded_img_url'],
      userId: json['userId'],
      disease: Disease.fromJson(json['diseaseId']), // Handling nested diseaseId
      createdAt: DateTime.parse(json['created_at']),
    );
  }
}
