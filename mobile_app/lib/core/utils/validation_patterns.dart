class ValidationPatterns {
  static final RegExp emailPattern = RegExp(
    r'^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$',
  );

  static final RegExp passwordPattern = RegExp(
    r'.{6,}',
  );

  static final RegExp namePattern = RegExp(
    r'^[a-zA-Z\s]+$',
  );

  static final RegExp phonePattern = RegExp(
    r'^\+?[0-9]{10,15}$',
  );

  static final RegExp urlPattern = RegExp(
    r'^(http|https)://[a-zA-Z0-9-\.]+\.[a-zA-Z]{2,}(/\S*)?$',
  );
}
