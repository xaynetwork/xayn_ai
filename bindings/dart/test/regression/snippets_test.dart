import 'package:flutter_test/flutter_test.dart' show group, test;

import 'package:xayn_ai_ffi_dart/src/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/reranker/ai.dart' show XaynAi;
import '../utils.dart' show model, vocab;

void main() {
  group('Regression', () {
    test('truncation length 90', () {
      final documents = [
        Document(
          '8821f73c-a866-11eb-9a4c-37402421f062',
          "by Zayn. 4.7 out of 5 stars 4. Audio CD Currently unavailable. Audio CD \$902.81 \$ 902. 81. \$3.99 shipping. Only 1 left in stock - order soon. N o w Remi es. by Various, Rag'N'Bone Man Bruno Mars Justin Bieber & BloodPop® feat. Julia Michaels Dua Lipa Rita Ora, et al. Audio CD \$902.81 \$ 902. 81.",
          13,
        ),
        Document(
          '8821f73f-a866-11eb-9a4c-37402421f062',
          "Early life. Zayn was born Zain Javadd Malik on 12 January 1993 in Bradford, West Yorkshire, England. His father, Yaser Malik, is a British Pakistani; his mother, Trisha Malik (née Brannan), is of English and Irish descent and had converted to Islam upon her marriage to Zayn's father. Malik has one older sister, Doniya, and two younger sisters, Waliyha and Safaa.",
          16,
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
          '88221e4f-a866-11eb-9a4c-37402421f062',
          '#ZaynMalik , #GigiHadid & #BellaHadid Grab Grilled Cheese Sandwiches Off A Food Truck For Her 26th Birthday In New York 4.23.21 - TheHollywoodFi [Video & Imagery Supplied By Getty Images] (Used Under License/With Permission) SUBSCRIBE To The Channel & Follow On Social Media: INSTAGRAM: http://www.Instagram.com/TheHollywoodFi TWITTER: https ...',
          36,
        ),
      ];

      final ai = XaynAi(vocab, model);
      ai.rerank([], documents);
      ai.free();
    });
  });
}
