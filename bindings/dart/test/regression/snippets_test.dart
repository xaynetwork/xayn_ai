import 'package:flutter_test/flutter_test.dart' show group, test;

import 'package:xayn_ai_ffi_dart/src/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/reranker/ai.dart' show XaynAi;
import '../utils.dart' show model, vocab;

void main() {
  group('Regression', () {
    test('snippets with different lengths and plenty of weird characters', () {
      final documents = [
        Document(
          '8821f730-a866-11eb-9a4c-37402421f062',
          "Xayn is the search engine alternative for people who cherish their privacy and don’t want to give up top-level convenience. We launched Xayn because we were tired of being stuck between a privacy-sucking search engine such as Google, yet the privacy alternatives out there just didn't cut it for us.",
          1,
        ),
        Document(
          '8821f731-a866-11eb-9a4c-37402421f062',
          "Xayn is the search engine alternative for people who cherish their privacy and don’t want to give up top-level convenience. We launched Xayn because we were tired of being stuck between a privacy-sucking search engine such as Google, yet the privacy alternatives out there just didn't cut it for us.",
          2,
        ),
        Document(
          '8821f732-a866-11eb-9a4c-37402421f062',
          'Xayn | 1,496 followers on LinkedIn. We build Web Search as it should be — Convenient, controllable, private. | Xayn is a privacy-protecting search alternative that enables users to gain back ...',
          3,
        ),
        Document(
          '8821f733-a866-11eb-9a4c-37402421f062',
          'What Xayn offers is the future of how we should approach search and data privacy. Xayn enables its users to control the search algorithms. Through a simple swipe on the results, users can change ...',
          4,
        ),
        Document(
          '8821f734-a866-11eb-9a4c-37402421f062',
          'Xayn does seem alive to the risk of the swiping mechanic resulting in the app feeling arduous. Lundbæk says the team is looking to add “some kind of gamification aspect” in the future — to ...',
          5,
        ),
        Document(
          '8821f735-a866-11eb-9a4c-37402421f062',
          'Xayn is a search engine that ushers in the new generation of user-friendly privacy tech. The app combines privacy, control over its algorithms, and convenience. It uses decentralized AI and trains directly on your device, leaving all your data with you.',
          6,
        ),
        Document(
          '8821f736-a866-11eb-9a4c-37402421f062',
          'Xayn is pitching itself, like Firefo and Opera, as the anti-Google, with a web browser and newsreader app that will protect your privacy but still offer tailored search results. The company says ...',
          7,
        ),
        Document(
          '8821f737-a866-11eb-9a4c-37402421f062',
          'Xayn introduces user-friendly and privacy-protecting web search. Currently, search engines deliver a profiled search e perience or an unprofiled-but-private search.',
          8,
        ),
        Document(
          '8821f738-a866-11eb-9a4c-37402421f062',
          'Welcome to the Zayn Malik Official Store! Pre-Order the debut album “Mind of Mine” on CD, Vinyl, or MP3 format. Shop online for official Zayn merchandise.',
          9,
        ),
        Document(
          '8821f739-a866-11eb-9a4c-37402421f062',
          'ZAYN Logo White Socks with Red Font \$30.00. ADD TO BASKET. NIL Button Pack \$15.00. ADD TO BASKET. NIL Patch Set \$40.00. ADD TO BASKET. NIL Skateboard Art \$250.00. ADD TO BASKET. NIL Sticker Set \$40.00. ADD TO BASKET. ZAYN Logo with Color Faces Phone Case \$35.00. ADD TO BASKET. ZAYN Tattoo Phone Case \$35.00.',
          10,
        ),
        Document(
          '8821f73a-a866-11eb-9a4c-37402421f062',
          '“Calamity” is the first track of ZAYN’s album, Nobody is Listening.It’s the only rap song on the album. Nobody can be too sure of what this song really means.',
          11,
        ),
        Document(
          '8821f73b-a866-11eb-9a4c-37402421f062',
          'Xayn is the 64,071 st most popular name of all time. How many people with the first name Xayn have been born in the United States? From 1880 to 2019, the Social Security Administration has recorded 19 babies born with the first name Xayn in the United States.',
          12,
        ),
        Document(
          '8821f73c-a866-11eb-9a4c-37402421f062',
          "by Zayn. 4.7 out of 5 stars 4. Audio CD Currently unavailable. Audio CD \$902.81 \$ 902. 81. \$3.99 shipping. Only 1 left in stock - order soon. N o w Remi es. by Various, Rag'N'Bone Man Bruno Mars Justin Bieber & BloodPop® feat. Julia Michaels Dua Lipa Rita Ora, et al. Audio CD \$902.81 \$ 902. 81.",
          13,
        ),
        Document(
          '8821f73d-a866-11eb-9a4c-37402421f062',
          'Singer-songwriter Ingrid Michaelson quickly got a glimpse into what it\'s like to have fame on the level of Zayn Malik and Gigi Hadid. The 41-year-old "Girls Chase Boys" performer briefly sent ...',
          14,
        ),
        Document(
          '8821f73e-a866-11eb-9a4c-37402421f062',
          "Zayn Teases a 'Run-In' During Mayweather vs. Paul YouTube sensation Logan Paul was present for Sami Zayn's match against Kevin Owens at WrestleMania 37, and Zayn may be looking to return the favor.",
          15,
        ),
        Document(
          '8821f73f-a866-11eb-9a4c-37402421f062',
          "Early life. Zayn was born Zain Javadd Malik on 12 January 1993 in Bradford, West Yorkshire, England. His father, Yaser Malik, is a British Pakistani; his mother, Trisha Malik (née Brannan), is of English and Irish descent and had converted to Islam upon her marriage to Zayn's father. Malik has one older sister, Doniya, and two younger sisters, Waliyha and Safaa.",
          16,
        ),
        Document(
          '8821f740-a866-11eb-9a4c-37402421f062',
          'Gigi Hadid and Zayn Malik got cheeky as they celebrated her 26th birthday on Friday, April 23, which marked her first birthday as a mom. The 28-year-old British singer and the model, who shares ...',
          17,
        ),
        Document(
          '8821f741-a866-11eb-9a4c-37402421f062',
          'Zayn talks quite a bit about his an iety, him losing weight while in One Direction because of the stress of touring etc and his decision to leave One Direction. I learnt that Zayn is quiet and prefers to be by himself sometimes, he even considers himself as an outsider. Zayn briefly talks about his relationship breaking down with Perrie.',
          18,
        ),
        Document(
          '8821f742-a866-11eb-9a4c-37402421f062',
          "ZAYN's debut album 'Mind Of Mine' out now. Get it on Apple Music: http://smarturl.it/MindOfMine?IQid=yt Target Delu e with 2 Bonus Tracks: http://smarturl.it...",
          19,
        ),
        Document(
          '8821f743-a866-11eb-9a4c-37402421f062',
          'In his book “ZAYN”, the singer wrote : “Malay and me were spending an afternoon sitting around the pool at the Beverly Hills Hotel, writing down lyrics and messing about with melodies.Malay ...',
          20,
        ),
        Document('88221e40-a866-11eb-9a4c-37402421f062', '', 21),
        Document('88221e41-a866-11eb-9a4c-37402421f062', '', 22),
        Document('88221e42-a866-11eb-9a4c-37402421f062', '', 23),
        Document('88221e43-a866-11eb-9a4c-37402421f062', '', 24),
        Document('88221e44-a866-11eb-9a4c-37402421f062', '', 25),
        Document('88221e45-a866-11eb-9a4c-37402421f062', '', 26),
        Document('88221e46-a866-11eb-9a4c-37402421f062', '', 27),
        Document('88221e47-a866-11eb-9a4c-37402421f062', '', 28),
        Document('88221e48-a866-11eb-9a4c-37402421f062', '', 29),
        Document('88221e49-a866-11eb-9a4c-37402421f062', '', 30),
        Document(
          '88221e4a-a866-11eb-9a4c-37402421f062',
          'After talking about his dancing skills, Sami Zayn connects with Paul Heyman over his inner turmoil. WWE action on Peacock, WWE Network, FOX, USA Network, Sony India and more. Stream WWE on Peacock https://pck.tv/3l4d8TP in the U.S. and on WWE Network http://wwe.yt/wwenetwork everywhere else ...',
          31,
        ),
        Document(
          '88221e4b-a866-11eb-9a4c-37402421f062',
          'Zayn Malik wins baby daddy of the year! Gigi Hadid just turned 26 and her boyfriend, Zayn, is not letting her birthday go unnoticed. The “Pillowtalk” singer wowed the model with a huge floral bouquet to celebrate the milestone, which features a variety of bright spring bulbs.',
          32,
        ),
        Document(
          '88221e4c-a866-11eb-9a4c-37402421f062',
          'Zayn Malik wins baby daddy of the year! Gigi Hadid just turned 26 and her boyfriend, Zayn, is not letting her birthday go unnoticed. The “Pillowtalk” singer wowed the model with a huge floral bouquet to celebrate the milestone, which features a variety of bright spring bulbs. » SUBSCRIBE: http://bit.ly/AHSub » Visit Our Website: http ...',
          33,
        ),
        Document(
          '88221e4d-a866-11eb-9a4c-37402421f062',
          "#BellaHadid Pulls #ZaynMalik 's Hand Off #GigiHadid 's Butt While Grabbing Grilled Cheese Sandwiches From A Food Truck On Her 26th Birthday In New York 4.23.21 - TheHollywoodFi [Video & Imagery Supplied By JosiahW/BACKGRID] (Used Under License/With Permission) SUBSCRIBE To The Channel & Follow On Social Media: INSTAGRAM: http://www.Instagram ...",
          34,
        ),
        Document(
          '88221e4e-a866-11eb-9a4c-37402421f062',
          'Apollo Crews puts his title on the line against KO with Commander Azeez lurking in the shadows and Sami Zayn on commentary. Watch WWE action on Peacock, WWE Network, FOX, USA Network, Sony India and more. #SmackDown Stream WWE on Peacock https://pck.tv/3l4d8TP in the U.S. and on WWE Network http://wwe.yt/wwenetwork everywhere else ...',
          35,
        ),
        Document(
          '88221e4f-a866-11eb-9a4c-37402421f062',
          '#ZaynMalik , #GigiHadid & #BellaHadid Grab Grilled Cheese Sandwiches Off A Food Truck For Her 26th Birthday In New York 4.23.21 - TheHollywoodFi [Video & Imagery Supplied By Getty Images] (Used Under License/With Permission) SUBSCRIBE To The Channel & Follow On Social Media: INSTAGRAM: http://www.Instagram.com/TheHollywoodFi TWITTER: https ...',
          36,
        ),
        Document(
            '88221e50-a866-11eb-9a4c-37402421f062', 'We see you, Zayn', 37),
        Document(
          '88221e51-a866-11eb-9a4c-37402421f062',
          'KO battles The Master Strategist in a WrestleMania rematch. Watch WWE action on Peacock, WWE Network, FOX, USA Network, Sony India and more. Stream WWE on Peacock https://pck.tv/3l4d8TP in the U.S. and on WWE Network http://wwe.yt/wwenetwork everywhere else --------------------------------------------------------------------- Follow WWE on ...',
          38,
        ),
        Document(
          '88221e52-a866-11eb-9a4c-37402421f062',
          'ZAYN ft. Sia - Dusk Till Dawn (Lyric Video) Song: Dusk Till Dawn ZAYN ft. Sia Discover the best pop music & chill songs: http://bit.ly/Lovelifelyrics Subscribe & Turn on the 🔔 to never miss a new video - - - - - - - - - - 👍Suggested Video: Imagine Dragons - Follow You: http://bit.ly/Imagine_Dragons_Follow_You Selfish Love - DJ Snake ...',
          39,
        ),
        Document(
          '88221e53-a866-11eb-9a4c-37402421f062',
          'For e clusive content, behind the scenes, and uncut album reactions: Patreon: https://bit.ly/2YnkBk7 -Follow Me On Social Media- TWITCH: http://www.twitch.tv/positivesteven Instagram: http://instagram.com/positivesteven Twitter: http://twitter.com/positivesteven My Spotify: https://spoti.fi/2SLyabN GoodReads: https://www.goodreads.com/user/show ...',
          40,
        ),
      ];

      final ai = XaynAi(vocab, model);
      ai.rerank([], documents);
      ai.free();
    });
  });
}
