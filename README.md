# Trading Signals Bot

Een Python-based trading signals bot die technische analyse uitvoert op cryptocurrency pairs en signalen verstuurt via n8n webhooks.

## Features

- Real-time monitoring van crypto pairs op Kraken
- Technische analyse met RSI en MACD indicators
- Automatische signaal generatie voor trading opportunities
- Webhook integratie met n8n voor signaal verwerking

## Ondersteunde Trading Pairs

- BTC/USD (XBT/USD op Kraken)
- ETH/USD
- XRP/USD
- SOL/USD
- DOT/USD

## Vereisten

- Python 3.8+
- Kraken API credentials
- n8n webhook URL

## Environment Variables

Maak een `.env` bestand aan met de volgende variabelen:

```env
KRAKEN_API_KEY=your_api_key
KRAKEN_API_SECRET=your_api_secret
WEBHOOK_URL=your_n8n_webhook_url
```

## Installatie

1. Clone de repository
2. Installeer dependencies: `pip install -r requirements.txt`
3. Voeg je environment variables toe
4. Start de bot: `python main.py`

## Deployment

Deze bot is geconfigureerd voor deployment op Railway.app:

1. Fork deze repository
2. Maak een nieuw project aan op Railway
3. Connect je GitHub repository
4. Voeg de environment variables toe in Railway
5. Deploy!
