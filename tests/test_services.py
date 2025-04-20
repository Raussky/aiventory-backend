import pytest
import base64
from datetime import datetime, timedelta
import io
import csv

from app.services.barcode import decode_barcode_from_base64, generate_barcode, verify_barcode
from app.services.file_parser import detect_and_parse_file, parse_csv, parse_excel
from app.services.prediction import PredictionService
from app.models.inventory import TimeFrame


@pytest.mark.asyncio
async def test_decode_barcode_from_base64(mock_barcode_service):
    """Test decoding barcode from base64 image"""
    # Since we're mocking the service, this just verifies the call pattern
    result = await decode_barcode_from_base64("base64_encoded_data")
    assert result == "1234567890123"
    mock_barcode_service.assert_called_once_with("base64_encoded_data")


@pytest.mark.asyncio
async def test_decode_barcode_from_base64_with_prefix(mock_barcode_service):
    """Test decoding barcode from base64 image with data URL prefix"""
    result = await decode_barcode_from_base64("data:image/png;base64,base64_encoded_data")
    assert result == "1234567890123"
    # Check that the function properly handled the data URL prefix
    mock_barcode_service.assert_called_once_with("base64_encoded_data")


@pytest.mark.asyncio
async def test_barcode_verification():
    """Test barcode verification logic"""
    # Test valid EAN-13 code
    valid_ean13 = "5901234123457"  # Example valid EAN-13
    result = await verify_barcode(valid_ean13)
    assert result is True

    # Test invalid EAN-13 code
    invalid_ean13 = "5901234123458"  # Changed last digit
    result = await verify_barcode(invalid_ean13)
    assert result is False

    # Test matching with product barcode
    product_barcode = "1234567890123"
    result = await verify_barcode("1234567890123", product_barcode)
    assert result is True

    # Test non-matching with product barcode
    result = await verify_barcode("9876543210123", product_barcode)
    assert result is False


@pytest.mark.asyncio
async def test_file_parser_detect_and_parse(mock_file_parser):
    """Test file type detection and parsing"""

    class MockFile:
        def __init__(self, filename):
            self.filename = filename
            self.content = b"mock content"

        async def read(self):
            return self.content

    # Test CSV detection
    csv_file = MockFile("test.csv")
    result = await detect_and_parse_file(csv_file)
    assert len(result) > 0
    assert "name" in result[0]

    # Test Excel detection
    xlsx_file = MockFile("test.xlsx")
    result = await detect_and_parse_file(xlsx_file)
    assert len(result) > 0
    assert "name" in result[0]

    # Test unsupported file type
    txt_file = MockFile("test.txt")
    with pytest.raises(ValueError):
        await detect_and_parse_file(txt_file)


@pytest.mark.asyncio
async def test_csv_parser():
    """Test CSV parser functionality"""
    # Create a test CSV file
    csv_content = io.StringIO()
    writer = csv.writer(csv_content)
    writer.writerow(["name", "category", "barcode", "quantity", "expire_date"])
    writer.writerow(["Test Product", "Test Category", "1234567890123", "10", "2025-05-01"])
    csv_content.seek(0)

    class MockFile:
        def __init__(self, content):
            self.content = content
            self.filename = "test.csv"

        async def read(self):
            return self.content.getvalue().encode()

    mock_file = MockFile(csv_content)

    # Parse the CSV
    result = await parse_csv(mock_file)
    assert len(result) == 1
    assert result[0]["name"] == "Test Product"
    assert result[0]["category"] == "Test Category"
    assert result[0]["barcode"] == "1234567890123"
    assert result[0]["quantity"] == "10"  # Still a string at this point
    assert result[0]["expire_date"] == "2025-05-01"


@pytest.mark.asyncio
async def test_prediction_service(db_session, test_product, test_sale):
    """Test prediction service functionality"""
    # Create a prediction service instance
    prediction_service = PredictionService(db_session)

    # Test forecast generation
    forecasts = await prediction_service.generate_forecast(
        product_sid=test_product.sid,
        timeframe=TimeFrame.MONTH,
        periods_ahead=3
    )

    # Since this is a stub implementation, we expect some results
    assert len(forecasts) == 3

    # Check structure of forecasts
    for forecast in forecasts:
        assert forecast["product_sid"] == test_product.sid
        assert forecast["timeframe"] == TimeFrame.MONTH
        assert "period_start" in forecast
        assert "period_end" in forecast
        assert "forecast_qty" in forecast
        assert "generated_at" in forecast
        assert "model_version" in forecast

    # Test saving forecasts to database
    saved_predictions = await prediction_service.save_forecast(forecasts)
    assert len(saved_predictions) == 3

    # Check database persistence
    for prediction in saved_predictions:
        assert prediction.product_sid == test_product.sid
        assert prediction.timeframe == TimeFrame.MONTH
        assert prediction.period_start is not None
        assert prediction.period_end is not None
        assert prediction.forecast_qty > 0
        assert prediction.generated_at is not None
        assert prediction.model_version is not None
        assert "stub" in prediction.model_version.lower()  # In the stub implementation


@pytest.mark.asyncio
async def test_prediction_service_insufficient_data(db_session):
    """Test prediction service behavior with insufficient data"""
    # Create a prediction service instance
    prediction_service = PredictionService(db_session)

    # Use a non-existent product SID
    non_existent_sid = "nonexistent_product_sid"

    # Try to generate forecast without sales data
    forecasts = await prediction_service.generate_forecast(
        product_sid=non_existent_sid,
        timeframe=TimeFrame.MONTH,
        periods_ahead=3
    )

    # The stub implementation should still produce some results
    # In a real implementation, this might return an empty list
    assert isinstance(forecasts, list)

    # If it returns results, they should have the correct structure
    if forecasts:
        for forecast in forecasts:
            assert forecast["product_sid"] == non_existent_sid
            assert forecast["timeframe"] == TimeFrame.MONTH