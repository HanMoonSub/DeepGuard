class BinaryClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if self.category == 1:
            sign = 1
        else:
            sign = -1
        return model_output * sign

class BinaryClassifierColumnTarget:
    """BinaryClassifierOutputTarget의 다중 출력 버전: (B,C) 출력 중 column 열의 logit을 category 부호로 사용한다."""

    def __init__(self, category, column=0):
        self.category = category
        self.column = column

    def __call__(self, model_output):
        sign = 1 if self.category == 1 else -1
        return model_output[..., self.column] * sign
