# Ethiopian E-commerce Data Extractor

A comprehensive data ingestion and preprocessing system for Ethiopian Telegram e-commerce channels. This system collects, processes, and stores messages from multiple Ethiopian e-commerce Telegram channels with special handling for Amharic text.

## Features

- **Multi-channel Data Collection**: Fetches data from 10+ Ethiopian e-commerce Telegram channels
- **Amharic Text Processing**: Specialized preprocessing for Amharic language features
- **Entity Extraction**: Extracts prices, phone numbers, hashtags, and mentions
- **Structured Storage**: SQLite database with organized data storage
- **Real-time Processing**: Handles text, images, and documents as they are posted
- **Comprehensive Logging**: Detailed logging for monitoring and debugging
- **Export Capabilities**: Export data in CSV, JSON, and pickle formats

## Selected Channels

The system is configured to collect data from the following Ethiopian e-commerce channels:

1. **ZemenExpress** - General e-commerce
2. **nevacomputer** - Computer and electronics
3. **meneshayeofficial** - Fashion and accessories
4. **ethio_brand_collection** - Branded products
5. **Leyueqa** - Various products
6. **sinayelj** - Electronics and gadgets
7. **Shewabrand** - Fashion and lifestyle
8. **helloomarketethiopia** - General marketplace
9. **modernshoppingcenter** - Shopping center
10. **qnashcom** - Electronics and more

## Prerequisites

- Python 3.8 or higher
- Telegram API credentials (API ID, API Hash, Phone Number)
- Git

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Amharic-E-commerce-Data-Extractor
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp env.example .env
   ```

5. **Configure Telegram API credentials**:
   Edit the `.env` file and add your Telegram API credentials:
   ```
   TELEGRAM_API_ID=your_api_id_here
   TELEGRAM_API_HASH=your_api_hash_here
   TELEGRAM_PHONE=your_phone_number_here
   ```

   To get these credentials:
   - Visit https://my.telegram.org/apps
   - Create a new application
   - Copy the API ID and API Hash
   - Use your phone number in international format

## Usage

### Quick Start

Run the complete data ingestion pipeline:

```bash
python scripts/data_ingestion_pipeline.py --mode full --limit 50 --days 7
```

### Command Line Options

```bash
python scripts/data_ingestion_pipeline.py [OPTIONS]

Options:
  --mode {full,incremental}  Pipeline mode: full or incremental update
  --limit INT                Maximum messages per channel (default: 100)
  --days INT                 Number of days back to fetch from (default: 30)
  --save-raw                 Save raw data to CSV files
  --stats-only               Only show current statistics
```

### Examples

1. **Full data collection** (100 messages per channel, last 30 days):
   ```bash
   python scripts/data_ingestion_pipeline.py --mode full
   ```

2. **Incremental update** (50 messages per channel, last 7 days):
   ```bash
   python scripts/data_ingestion_pipeline.py --mode incremental --limit 50 --days 7
   ```

3. **View current statistics**:
   ```bash
   python scripts/data_ingestion_pipeline.py --stats-only
   ```

4. **Save raw data**:
   ```bash
   python scripts/data_ingestion_pipeline.py --mode full --save-raw
   ```

## Project Structure

```
Amharic-E-commerce-Data-Extractor/
├── src/
│   ├── core/
│   │   ├── telegram_client.py      # Telegram API client
│   │   ├── data_preprocessor.py    # Amharic text preprocessing
│   │   └── data_storage.py         # Database and file storage
│   ├── services/                   # Business logic services
│   ├── models/                     # Data models
│   └── utils/                      # Utility functions
├── scripts/
│   └── data_ingestion_pipeline.py  # Main pipeline script
├── config/
│   └── settings.py                 # Configuration settings
├── data/                           # Data storage directory
│   ├── raw/                        # Raw data files
│   ├── processed/                  # Processed data files
│   ├── exports/                    # Exported data
│   └── backups/                    # Database backups
├── logs/                           # Log files
├── tests/                          # Test files
├── requirements.txt                # Python dependencies
├── env.example                     # Environment variables template
└── README.md                       # This file
```

## Data Processing Pipeline

### 1. Data Ingestion
- Connects to Telegram API using Telethon
- Fetches messages from selected channels
- Handles rate limiting and error recovery
- Collects metadata (views, forwards, replies, media info)

### 2. Text Preprocessing
- **Language Detection**: Identifies Amharic, English, and mixed text
- **Text Cleaning**: Removes URLs, normalizes whitespace, handles emojis
- **Amharic Normalization**: Handles Amharic-specific character variations
- **Entity Extraction**: Extracts prices, phone numbers, hashtags, mentions

### 3. Data Storage
- **SQLite Database**: Structured storage with indexed tables
- **File Storage**: CSV, JSON, and pickle exports
- **Metadata Tracking**: Timestamps, processing statistics, channel information

### 4. Data Export
- **Multiple Formats**: CSV, JSON, pickle
- **Filtered Exports**: By language, channel, date range
- **Statistics**: Comprehensive data analytics

## Amharic Text Processing

The system includes specialized processing for Amharic text:

- **Character Detection**: Identifies Amharic Unicode ranges
- **Currency Extraction**: Ethiopian Birr (ብር, birr, ETB)
- **Phone Number Detection**: Ethiopian phone number patterns
- **E-commerce Terms**: Amharic e-commerce vocabulary
- **Text Normalization**: Handles Amharic diacritics and variations

## Database Schema

### Messages Table
- `message_id`: Unique message identifier
- `channel_username`: Source channel
- `text`: Original message text
- `text_cleaned`: Preprocessed text
- `text_normalized`: Normalized Amharic text
- `language`: Detected language
- `contains_amharic`: Boolean flag
- `is_ecommerce`: E-commerce message flag
- `prices`: Extracted prices (JSON)
- `phone_numbers`: Extracted phone numbers (JSON)
- `hashtags`: Extracted hashtags (JSON)
- `mentions`: Extracted mentions (JSON)
- `urls`: Extracted URLs (JSON)
- `date`: Message timestamp
- `views`, `forwards`, `replies`: Engagement metrics

### Channels Table
- `username`: Channel username
- `title`: Channel title
- `description`: Channel description
- `participants_count`: Number of participants

### Preprocessing Stats Table
- `run_date`: Processing run timestamp
- `total_messages`: Total messages processed
- `messages_with_amharic`: Amharic messages count
- `ecommerce_messages`: E-commerce messages count
- `languages`: Language distribution (JSON)
- `channels`: Channel distribution (JSON)

## Configuration

Edit `config/settings.py` to customize:

- **Channel Selection**: Choose which channels to monitor
- **Data Collection**: Limit messages per channel, time range
- **Preprocessing**: Text cleaning options, entity extraction
- **Storage**: Database path, backup settings
- **Performance**: Batch sizes, worker counts

## Monitoring and Logging

The system provides comprehensive logging:

- **File Logging**: Detailed logs in `logs/data_ingestion.log`
- **Console Output**: Real-time progress updates
- **Error Tracking**: Failed messages and processing errors
- **Performance Metrics**: Processing times and throughput

## Data Export

Export data in various formats:

```python
from src.core.data_storage import DataStorage

storage = DataStorage()

# Export to CSV
storage.export_data(format='csv')

# Export to JSON
storage.export_data(format='json')

# Export to pickle
storage.export_data(format='pickle')
```

## Error Handling

The system includes robust error handling:

- **Rate Limiting**: Automatic delays between requests
- **Retry Logic**: Failed requests are retried
- **Graceful Degradation**: Continues processing on individual failures
- **Error Logging**: Detailed error tracking and reporting

## Performance Considerations

- **Batch Processing**: Processes messages in batches
- **Rate Limiting**: Respects Telegram API limits
- **Memory Management**: Efficient data structures
- **Database Indexing**: Optimized queries

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:

1. Check the logs in `logs/data_ingestion.log`
2. Review the configuration in `config/settings.py`
3. Ensure Telegram API credentials are correct
4. Check network connectivity

## Future Enhancements

- **Real-time Streaming**: Live message processing
- **Advanced NLP**: Sentiment analysis, topic modeling
- **Image Processing**: OCR for product images
- **API Endpoints**: REST API for data access
- **Dashboard**: Web-based monitoring interface
- **Machine Learning**: Automated entity extraction models
