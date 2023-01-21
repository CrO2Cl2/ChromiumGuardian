# ChromiumGuardain [CURRENTLY BROKEN]
 spam detection algorythm made for the usage with discord bots. The Database was trained from actual discord messages. The algorythm can aslo detect https and http urls,  discord invite links, and gifs fron tenor
 All the discord messages have been anonymized and no perosnal information is contained. All the messages have been collected with the consent of the original author.

 Usage: 
  put the spamdec.py file in the same directory as your discord bot. then import the file into your project, when inputting spamdec.classify_message(message), it will return  "spam", "not spam", "discord_invite", "tenor_gif" or "url" you then will be able to use an if clause to make the actions that you want on different messages

 DISCLAMER:
 this is still a crude prototype. It doesn't output a lot of false positives, but it does output a lot of false negatives

 HOW DO I HELP?

if you want to help my effort to reduce spam, you can help by adding entries to the database. I would be very thankful for your contribution. Please note to input only german or English messages, and to correctly label them. I included a handy little tool to add entires faster.
Please note to not add any personal or harmful information to this database. Also note that any NSFW messages are not desiered. 
ADD THE MESSAGES ONLY IF YOU HAVE THE OP'S PERMISSION TO DO SO AND HAVE THE PERMISSION FROM THE SERVER OWNER.

