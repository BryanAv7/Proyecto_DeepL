import { bootstrapApplication } from '@angular/platform-browser';
import { AppComponent } from './app/app.component';  
import { appConfig } from './app/app.config';
import { initializeApp } from 'firebase/app';
import { environment } from './environments/environment';

initializeApp(environment.firebaseConfig);

bootstrapApplication(AppComponent, appConfig)
  .catch(err => console.error(err));
