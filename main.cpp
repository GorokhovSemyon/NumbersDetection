#include <opencv2/opencv.hpp>
#include <opencv2/text.hpp>

int main()
{
    // Загрузка изображения
    cv::Mat image = cv::imread("../imgs/test_marker.jpg", cv::IMREAD_COLOR);
    cv::Mat src = cv::imread("../imgs/test_marker.jpg", cv::IMREAD_COLOR);

    // Применение размытия для сглаживания мелких шумов
    cv::medianBlur(image, image, 9);

    // Преобразование изображения в оттенки серого
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // Применение адаптивного порогового преобразования для выделения темных областей
    cv::Mat thresholded;
    cv::adaptiveThreshold(grayImage, thresholded, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 11, 2);

    // Поиск контуров темных областей
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresholded, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Создание прямоугольников для каждого контура с ограничением минимальной площади и наличием цифр
    std::vector<cv::Rect> rectangles;
    cv::Ptr<cv::text::OCRTesseract> ocr = cv::text::OCRTesseract::create();
    for (const auto& contour : contours)
    {
        // Получение ограничивающего прямоугольника
        cv::Rect boundingRect = cv::boundingRect(contour);

        // Ограничения площади
        float area = boundingRect.width * boundingRect.height;
        if (area >= 1000 && area <= 100000) // Варьируемые параметры площади
        {
            // Распознавание текста в прямоугольнике
            cv::Mat roi = thresholded(boundingRect);
            std::string text;
            ocr->run(roi, text);

            // Проверка наличия цифр
            if (std::any_of(text.begin(), text.end(), ::isdigit))
            {
                rectangles.push_back(boundingRect);
            }
        }
    }

    // Отображение прямоугольников на исходном изображении
    for (const auto& rect : rectangles)
    {
        cv::rectangle(src, rect, cv::Scalar(0, 255, 0), 2);
    }

    // Отображение изображения с выделенными прямоугольниками
    cv::imshow("Result", src);
    cv::waitKey(0);

    return 0;
}