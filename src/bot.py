import asyncio
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from .utils import setup_logging, get_env_var
from .rag import RAGSystem
from .vector_db import ChromaDB

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

class TPPBot:
    def __init__(self):
        self.token = get_env_var("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set")

        self.application = Application.builder().token(self.token).build()
        self.rag_system = RAGSystem()
        self.chroma_db = ChromaDB()

        # Collection mapping for user-friendly names
        self.collection_names = {
            "all": "Все коллекции",
            "719": "Консультации по 719-ПП",
            "support": "Поддержка бизнеса",
            "services": "Услуги ТПП",
            "membership": "Членство в ТПП",
            "events": "Мероприятия",
            "cooperation": "Кооперация",
            "site": "Общий контент"
        }

        self.setup_handlers()

    def setup_handlers(self):
        """Setup bot handlers"""
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("collections", self.collections_command))
        self.application.add_handler(CallbackQueryHandler(self.handle_callback))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome_text = (
            "🤖 <b>ТПП УР AI-Помощник</b>\n\n"
            "Я интеллектуальный помощник Торгово-Промышленной Палаты Удмуртской Республики.\n\n"
            "Я могу ответить на вопросы о:\n"
            "• Услугах ТПП\n"
            "• Мерах поддержки бизнеса\n"
            "• Мероприятиях и обучении\n"
            "• Членстве в ТПП\n"
            "• Кооперации и партнёрстве\n\n"
            "Просто задайте вопрос, и я найду информацию на сайте udmtpp.ru!\n\n"
            "Используйте /help для справки."
        )

        keyboard = [
            [InlineKeyboardButton("📚 Коллекции", callback_data="collections")],
            [InlineKeyboardButton("❓ Помощь", callback_data="help")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(welcome_text, parse_mode='HTML', reply_markup=reply_markup)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = (
            "📖 <b>Справка по использованию</b>\n\n"
            "<b>Как задавать вопросы:</b>\n"
            "• Просто напишите вопрос на русском языке\n"
            "• Чем конкретнее вопрос, тем точнее ответ\n"
            "• Можно уточнить тему через /collections\n\n"
            "<b>Примеры вопросов:</b>\n"
            "• Какие услуги предоставляет ТПП?\n"
            "• Как вступить в ТПП?\n"
            "• Какие меры поддержки бизнеса?\n\n"
            "<b>Команды:</b>\n"
            "/start - начать работу\n"
            "/help - эта справка\n"
            "/collections - выбрать тему\n\n"
            "<i>Ответы формируются только на основе проверенной информации с сайта udmtpp.ru</i>"
        )

        await update.message.reply_text(help_text, parse_mode='HTML')

    async def collections_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /collections command"""
        await self.show_collections(update.message)

    async def show_collections(self, message):
        """Show collections selection"""
        collections_info = self.chroma_db.get_collection_info()

        text = "📚 <b>Выберите тему для поиска:</b>\n\n"

        keyboard = []
        for coll_id, coll_data in collections_info.items():
            name = self.collection_names.get(coll_id, coll_data['description'])
            count = coll_data['points_count']
            button_text = f"{name} ({count})"
            keyboard.append([InlineKeyboardButton(button_text, callback_data=f"select_collection_{coll_id}")])

        keyboard.append([InlineKeyboardButton("🔍 Все темы", callback_data="select_collection_all")])
        keyboard.append([InlineKeyboardButton("❌ Отмена", callback_data="cancel")])

        reply_markup = InlineKeyboardMarkup(keyboard)
        await message.reply_text(text, parse_mode='HTML', reply_markup=reply_markup)

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries"""
        query = update.callback_query
        await query.answer()

        data = query.data

        if data == "collections":
            await self.show_collections(query.message)
        elif data == "help":
            await self.help_command(update, context)
        elif data.startswith("select_collection_"):
            collection = data.replace("select_collection_", "")
            if collection == "all":
                collection = None
                collection_name = "всех коллекций"
            else:
                collection_name = self.collection_names.get(collection, collection)

            context.user_data['selected_collection'] = collection
            await query.message.edit_text(
                f"✅ Выбрана тема: <b>{collection_name}</b>\n\n"
                "Теперь задайте свой вопрос!",
                parse_mode='HTML'
            )
        elif data == "cancel":
            await query.message.edit_text("❌ Действие отменено")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages"""
        user_message = update.message.text.strip()
        user_id = update.effective_user.id

        # Show typing indicator
        await update.message.chat.send_action("typing")

        try:
            # Get selected collection
            selected_collection = context.user_data.get('selected_collection')

            # Process query
            logger.info(f"User {user_id} asked: {user_message[:50]}...")

            result = self.rag_system.ask(user_message, selected_collection)

            # Format response
            response_text = result['response']

            # Add confidence info if low
            if result['confidence'] < 0.7:
                response_text += f"\n\n⚠️ Уверенность ответа: {result['confidence']:.1%}"

            # Add sources if available
            if result['sources']:
                sources_text = "\n\n📄 <b>Источники:</b>\n"
                for i, source in enumerate(result['sources'][:3], 1):  # Limit to 3 sources
                    sources_text += f"{i}. {source['url']} (релевантность: {source['score']:.1%})\n"
                response_text += sources_text

            # Split long messages
            if len(response_text) > 4000:
                chunks = [response_text[i:i+4000] for i in range(0, len(response_text), 4000)]
                for chunk in chunks:
                    await update.message.reply_text(chunk, parse_mode='HTML')
            else:
                await update.message.reply_text(response_text, parse_mode='HTML')

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_text = (
                "❌ Произошла ошибка при обработке запроса.\n"
                "Попробуйте переформулировать вопрос или обратитесь позже."
            )
            await update.message.reply_text(error_text)

    async def run(self):
        """Start the bot"""
        logger.info("Starting Telegram bot...")
        await self.application.initialize()
        await self.application.start()
        logger.info("Bot started successfully")

        # Keep the bot running
        await self.application.updater.start_polling()
        await self.application.updater.wait_for_shutdown()

def main():
    """Main entry point"""
    try:
        bot = TPPBot()
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot failed to start: {e}")
        raise

if __name__ == "__main__":
    main()
