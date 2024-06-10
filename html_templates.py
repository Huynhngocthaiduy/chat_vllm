css ="""
    <style>
        /* User Chat Message */

        .st-emotion-cache-janbn0 {
            background-color: #ffffff;
            
        }

        /* AI Chat Message */
    
        .st-emotion-cache-4oy321 {
            background-color: #f3f3f3;
            
        }

        section[data-testid="stSidebar"] {
            width: 300px !important;
        }
/* Background Image */
[data-testid="stAppViewContainer"] > .main {
    position: relative;
    z-index: 1; /* Ensure the content is on top of the background */
}

[data-testid="stAppViewContainer"] > .main::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 300px;
    height:200px;
    background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQSP7fVKbzilVR2RC1IuqoubRo4NOKENnJtG_2Fj4vB&s");
    background-size: contain;
    background-position: center;
    background-repeat: no-repeat;
    opacity: 0.9; /* Set the opacity level */
    z-index: 0; /* Place the background behind the content */}
    </style>
    """
    